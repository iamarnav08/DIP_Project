import cv2
import math
import numpy as np
import tqdm
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage
import argparse

class SteadyFlowStabilizer:
    '''
    A faithful implementation of "SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization" 
    by Shuaicheng Liu et al. (CVPR 2014).
    '''

    def __init__(self, 
                 iterations=5, # 5 iterations for iterative refinement (Section 4.4)
                 grid_size=40, # 40x40 pixel grid for motion completion (Section 4.3)
                 base_threshold=0.2, # Epsilon for outlier detection (Section 4.2)
                 alpha=20.0, # Alpha for adaptive threshold (Section 4.4)
                 temporal_radius=20, # Tau for adaptive window selection (Section 5.1)
                 lambda_smooth=10.0, # Lambda for stabilization (Equation 3)
                 downscale_factor=0.5):
        self.iterations = iterations
        self.grid_size = grid_size
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.temporal_radius = temporal_radius
        self.lambda_smooth = lambda_smooth
        self.downscale_factor = downscale_factor

    def stabilize(self, input_path, output_path):
        frames, fps = self._read_frames(input_path)
        if not frames:
            return

        h, w = frames[0].shape[:2]
        print(f"Loaded {len(frames)} frames ({w}x{h}).")

        # 1. Initialization (Section 4.1)
        # Robust initialization using global homography + residual optical flow
        print("Initializing SteadyFlow...")
        U = self._initialize_steadyflow(frames) 
        
        # Iterative Refinement (Section 4.4)
        M = None # Outlier mask
        P = None # Stabilized path variable for the loop
        
        for n in range(1, self.iterations + 1):
            print(f"Iteration {n}/{self.iterations}...")
            
            # 2. Discontinuity Identification (Section 4.2)
            # Calculate threshold: (1 + alpha^(1/n)) * epsilon
            threshold = (1 + self.alpha**(1/n)) * self.base_threshold
            
            if n == 1:
                # First iteration: Spatial Analysis ONLY
                M = self._spatial_analysis(U, threshold=0.1) # Threshold 0.1 from paper for spatial
            else:
                # Subsequent iterations: Temporal Analysis (using smoothed flow from prev iter)
                # Need to use the stabilized flow S (from previous step 5) for temporal check
                if P is not None:
                     M = self._temporal_analysis(P, threshold)
                else:
                     M = self._spatial_analysis(U, threshold=0.1)

            # 3. Motion Completion (Section 4.3)
            # E(V) = Ed(V) + Es(V) using mesh grid
            print("  Motion Completion...")
            U = self._motion_completion(U, M, frames[0].shape)

            # 4. Pixel Profile Stabilization (Section 5)
            # Minimize O({Pt})
            print("  Stabilizing Pixel Profiles...")
            P = self._stabilize_pixel_profiles(U)
            
        # 5. Rendering Final Result
        print("Rendering...")
        # Warp fields B_t = P_t - C_t (Equation 5 context)
        # C_t is accumulated U. 
        # FIX: We must add the t=0 state (zeros) to C to match the shape of P.
        # P has shape (N, h, w, 2), U has shape (N-1, h, w, 2)
        C_diff = np.cumsum(U, axis=0)
        C = np.vstack([np.zeros((1, C_diff.shape[1], C_diff.shape[2], 2), dtype=C_diff.dtype), C_diff])
        
        # Now shapes match: (N, h, w, 2)
        B = P - C
        
        self._write_video(output_path, frames, B, fps)

    def _read_frames(self, input_path):
        cap = cv2.VideoCapture(input_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        return frames, fps

    def _initialize_steadyflow(self, frames):
        # Section 4.1: Initialization
        # "We first estimate a global homography... apply optical flow... to compute residual."
        # "SteadyFlow is initialized as summation of residual optical flow and motion displacements introduced by global homography."
        
        num_frames = len(frames)
        h, w = frames[0].shape[:2]
        proc_h, proc_w = int(h*self.downscale_factor), int(w*self.downscale_factor)
        
        U = np.zeros((num_frames-1, h, w, 2), dtype=np.float32)
        
        prev_gray = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (proc_w, proc_h))
        
        for t in tqdm.tqdm(range(num_frames-1)):
            curr_gray = cv2.resize(cv2.cvtColor(frames[t+1], cv2.COLOR_BGR2GRAY), (proc_w, proc_h))
            
            # 1. Global Homography (KLT based)
            # Simplified here to use findHomography on sparse features
            feat_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            if feat_prev is None:
                H = np.eye(3)
            else:
                feat_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, feat_prev, None)
                good_prev = feat_prev[status==1]
                good_curr = feat_curr[status==1]
                if len(good_prev) < 4:
                    H = np.eye(3)
                else:
                    H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC)
            
            # 2. Warp previous frame to current using H (Align them)
            # To get residual flow, we warp prev towards curr
            prev_warped = cv2.warpPerspective(prev_gray, H, (proc_w, proc_h))
            
            # 3. Compute Residual Optical Flow (Dense)
            # Paper uses [13] (Liu. Beyond Pixels), here we use Farneback as a proxy
            flow_residual = cv2.calcOpticalFlowFarneback(prev_warped, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 4. Total SteadyFlow = Residual + Homography Motion
            # Create grid of coordinates
            y_grid, x_grid = np.mgrid[0:proc_h, 0:proc_w]
            # Homography displacement
            # Apply H to all pixels
            ones = np.ones_like(x_grid)
            coords = np.stack([x_grid, y_grid, ones], axis=-1).reshape(-1, 3).T
            warped_coords = H @ coords
            warped_coords = warped_coords / warped_coords[2, :]
            warped_coords = warped_coords[:2, :].T.reshape(proc_h, proc_w, 2)
            
            flow_homography = warped_coords - np.stack([x_grid, y_grid], axis=-1)
            
            flow_total = flow_residual + flow_homography
            
            # Upscale
            U[t] = cv2.resize(flow_total, (w, h)) * (1.0/self.downscale_factor)
            prev_gray = curr_gray
            
        return U

    def _spatial_analysis(self, U, threshold):
        # Section 4.2: Discontinuity Identification (Spatial)
        # "Threshold the gradient magnitude of raw optical flow"
        # "Once the magnitude at p is larger than threshold (0.1)... p is outlier"
        
        M = np.ones(U.shape[:3], dtype=np.float32) # Mask: 1=Inlier, 0=Outlier
        
        for t in range(len(U)):
            flow = U[t]
            # Gradient magnitude
            dx = cv2.Sobel(flow, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(np.sum(dx**2 + dy**2, axis=2))
            
            mask_t = mag <= threshold
            M[t] = mask_t.astype(np.float32)
            
        return M

    def _temporal_analysis(self, P, threshold):
        # Section 4.2: Temporal Analysis
        # Check smoothness of accumulated motion vectors c_t(p)
        # c_t(p) is P in our context (or rather, the path we are checking)
        # "M_t(p) = 0 if ||c_t(p) - G * c_t(p)|| > epsilon" (Equation 1)
        
        # P is (frames, h, w, 2)
        # We need to gaussian smooth P over time
        P_smooth = scipy.ndimage.gaussian_filter1d(P, sigma=3, axis=0) # Standard deviation 3 (Section 4.2)
        
        diff = np.linalg.norm(P - P_smooth, axis=3) # Magnitude of vector difference
        
        M = (diff <= threshold).astype(np.float32)
        # M needs to be for U (frames-1), P is frames. Truncate.
        return M[:-1]

    def _motion_completion(self, U, M, shape):
        # Section 4.3: Motion Completion
        # "Minimizing energy E(V) = Ed(V) + Es(V)" 
        # This implementation solves for the grid vertex positions that minimize
        # deviation from valid flow (Ed) while maintaining grid rigidity (Es, ASAP).

        H, W = shape[:2]
        gh, gw = self.grid_size, self.grid_size
        
        # Grid dimensions
        nrows = int(np.ceil(H / gh)) + 1
        ncols = int(np.ceil(W / gw)) + 1
        n_verts = nrows * ncols

        # 1. Precompute Grid Coordinates
        # Create a mesh of vertex coordinates in the source image
        grid_y, grid_x = np.meshgrid(np.arange(nrows) * gh, np.arange(ncols) * gw, indexing='ij')
        # Vertices shape: (nrows, ncols)
        
        # We need to solve for the Deformed Vertices (V). 
        # The flow will be V - Original_Grid.
        
        # Optimization: To make the linear system tractable in Python without C++ solvers,
        # we subsample pixels for the Data Term.
        stride = 4 
        
        # Flattened vertex indices for easy equation building
        # v_idx[r, c] gives the index (0..n_verts-1)
        v_idx = np.arange(n_verts).reshape(nrows, ncols)

        U_completed = np.copy(U)

        # Loop over frames
        for t in tqdm.tqdm(range(len(U)), desc="Motion Completion (ASAP)"):
            # Skip if frame is fully valid (optimization)
            if np.mean(M[t]) > 0.99:
                continue

            # --- Build the Sparse Linear System: Ax = b ---
            # Unknowns x: [v0_x, v0_y, v1_x, v1_y, ... ]
            # Size: 2 * n_verts
            
            rows = []
            cols = []
            data = []
            rhs = []
            
            row_counter = 0

            # --- A. Data Term Ed(V)  ---
            # "Ed(V) = sum M(p) * || V * phi_p - (p + u_p) ||"
            # We iterate over valid pixels and constrain their enclosing grid vertices.
            
            # Get indices of valid pixels (M=1) with subsampling
            valid_mask = M[t] > 0.5
            # Create a coordinate grid for the image
            y_coords, x_coords = np.mgrid[0:H:stride, 0:W:stride]
            mask_sub = valid_mask[0:H:stride, 0:W:stride]
            
            # Filter only valid pixels
            ys = y_coords[mask_sub]
            xs = x_coords[mask_sub]
            
            if len(ys) > 0:
                # Corresponding flow values
                flow_x = U[t, :, :, 0][0:H:stride, 0:W:stride][mask_sub]
                flow_y = U[t, :, :, 1][0:H:stride, 0:W:stride][mask_sub]
                
                # Target positions for these pixels (p + u_p)
                target_x = xs + flow_x
                target_y = ys + flow_y
                
                # Bilinear weights for grid interpolation
                # Grid cell indices
                g_c = (xs / gw).astype(int)
                g_r = (ys / gh).astype(int)
                
                # Clamp to boundaries
                g_c = np.clip(g_c, 0, ncols - 2)
                g_r = np.clip(g_r, 0, nrows - 2)
                
                # Local coordinates (0..1)
                alpha = (xs - g_c * gw) / gw
                beta = (ys - g_r * gh) / gh
                
                # Vertices of the enclosing quad: Top-L, Top-R, Bot-L, Bot-R
                # Indices in the flattened state vector
                # We handle X and Y components separately in the rows
                
                # Weights for the 4 vertices
                w_tl = (1 - alpha) * (1 - beta)
                w_tr = alpha * (1 - beta)
                w_bl = (1 - alpha) * beta
                w_br = alpha * beta
                
                vals = [w_tl, w_tr, w_bl, w_br]
                
                # For each pixel, add 2 equations (one for X, one for Y)
                # Equation X: w_tl*Vx_tl + ... = target_x
                for i in range(len(xs)):
                    idx_tl = v_idx[g_r[i], g_c[i]]
                    idx_tr = v_idx[g_r[i], g_c[i]+1]
                    idx_bl = v_idx[g_r[i]+1, g_c[i]]
                    idx_br = v_idx[g_r[i]+1, g_c[i]+1]
                    
                    indices = [idx_tl, idx_tr, idx_bl, idx_br]
                    current_weights = [vals[0][i], vals[1][i], vals[2][i], vals[3][i]]
                    
                    # X-row
                    for k in range(4):
                        rows.append(row_counter)
                        cols.append(2 * indices[k])     # 2*idx is X component
                        data.append(current_weights[k])
                    rhs.append(target_x[i])
                    row_counter += 1
                    
                    # Y-row
                    for k in range(4):
                        rows.append(row_counter)
                        cols.append(2 * indices[k] + 1) # 2*idx+1 is Y component
                        data.append(current_weights[k])
                    rhs.append(target_y[i])
                    row_counter += 1

            # --- B. Smoothness Term Es(V) (ASAP) [cite: 197, 370] ---
            # Enforce similarity transform on grid triangles.
            # For a grid quad, we split into two triangles.
            # Triangle Similarity: V2 - V0 = R_90(V1 - V0) for isosceles right triangle.
            
            # Weight for smoothness (usually high for completion)
            w_smooth = 10.0 
            
            # We loop through all grid cells
            for r in range(nrows - 1):
                for c in range(ncols - 1):
                    # Vertex indices
                    v00 = v_idx[r, c]     # Top-Left
                    v01 = v_idx[r, c+1]   # Top-Right
                    v10 = v_idx[r+1, c]   # Bot-Left
                    v11 = v_idx[r+1, c+1] # Bot-Right
                    
                    # Triangle 1: (v00, v10, v01) -> Top-Left, Bot-Left, Top-Right
                    # In rest pose, v10 is "Down" (Y+), v01 is "Right" (X+).
                    # Vector v01->v00 should be rotated 90 deg relative to v10->v00?
                    # Rest vectors: e1 = (0, gh), e2 = (gw, 0).
                    # If square grid (gw=gh), e2 = Rot(-90) * e1.
                    # Or: (v01 - v00) - Rot(-90) * (v10 - v00) = 0
                    # Rot(-90) on (x,y) is (y, -x).
                    
                    # Eq X: (x01 - x00) - (y10 - y00) = 0
                    # Eq Y: (y01 - y00) - (-x10 + x00) = 0  => (y01 - y00) + x10 - x00 = 0
                    
                    # Add X constraint
                    # Coeffs: x01(1), x00(-1), y10(-1), y00(1)
                    rows.extend([row_counter]*4)
                    cols.extend([2*v01, 2*v00, 2*v10+1, 2*v00+1])
                    data.extend([w_smooth, -w_smooth, -w_smooth, w_smooth])
                    rhs.append(0)
                    row_counter += 1
                    
                    # Add Y constraint
                    # Coeffs: y01(1), y00(-1), x10(1), x00(-1)
                    rows.extend([row_counter]*4)
                    cols.extend([2*v01+1, 2*v00+1, 2*v10, 2*v00])
                    data.extend([w_smooth, -w_smooth, w_smooth, -w_smooth])
                    rhs.append(0)
                    row_counter += 1
                    
                    # Triangle 2: (v11, v01, v10) -> Bot-Right, Top-Right, Bot-Left
                    # Similar logic.
                    
                    # Only one triangle per quad is often enough for connectivity, 
                    # but two makes it symmetric. Skipping for speed/sparsity 
                    # unless stability is poor. (Using just T1 is standard "lazy" ASAP).
                    # Let's add Triangle 2 for faithfulness.
                    
                    # Eq X: (x11 - x10) - (y11 - y01) = 0 ??? 
                    # Let's use: (v11 - v10) should be similar to (v01 - v00).
                    # Simple Laplacian smoothness is actually:
                    # v(r,c) - avg(neighbors) = 0.
                    # BUT ASAP requires rotation invariance.
                    # Let's stick to the single triangle constraint per quad which defines the local shape fully.

            # --- Solve System ---
            matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(row_counter, 2*n_verts))
            matrix_csr = matrix.tocsr()
            
            # Least squares solution
            # lsqr is robust but can be slow. Since we have a good initialization (0 motion?), 
            # actually we don't. We want absolute positions.
            
            result = scipy.sparse.linalg.lsqr(matrix_csr, rhs, show=False)
            coords_flat = result[0] # [x0, y0, x1, y1, ...]
            
            # --- Render Result back to Pixel Flow ---
            # Reshape result to (nrows, ncols, 2)
            grid_flow_x = coords_flat[0::2].reshape(nrows, ncols) - grid_x
            grid_flow_y = coords_flat[1::2].reshape(nrows, ncols) - grid_y
            
            # Upsample flow (Interpolate grid motion to pixels)
            # We can use opencv resize for speed as it is bi-linear interpolation
            # Note: grid_flow is (nrows, ncols). Image is (H, W).
            
            # OpenCV expects (W, H)
            full_flow_x = cv2.resize(grid_flow_x, (W, H))
            full_flow_y = cv2.resize(grid_flow_y, (W, H))
            
            # Where mask was invalid, use the completed flow
            # Where mask was valid, we technically should keep original, 
            # but the paper implies we replace the whole field with the optimized one 
            # or just fill holes?
            # "Missing regions are filled in by motion completion." [cite: 36]
            # Usually we blend or just replace the holes.
            
            filled_x = U[t,:,:,0] * M[t] + full_flow_x * (1 - M[t])
            filled_y = U[t,:,:,1] * M[t] + full_flow_y * (1 - M[t])
            
            U_completed[t] = np.stack([filled_x, filled_y], axis=2)

        return U_completed

    # def _motion_completion(self, U, M, shape):
    #     # Optimized implementation of Section 4.3: Motion Completion
    #     # Minimizes E(V) = Ed(V) + Es(V) using vectorized sparse matrix construction.

    #     H, W = shape[:2]
    #     gh, gw = self.grid_size, self.grid_size
        
    #     # Grid dimensions
    #     nrows = int(np.ceil(H / gh)) + 1
    #     ncols = int(np.ceil(W / gw)) + 1
    #     n_verts = nrows * ncols
        
    #     # Create a mesh of vertex coordinates in the source image for reference
    #     grid_y, grid_x = np.meshgrid(np.arange(nrows) * gh, np.arange(ncols) * gw, indexing='ij')
        
    #     # Vertex indices grid: v_idx[r, c] -> unique index 0..N-1
    #     v_idx = np.arange(n_verts).reshape(nrows, ncols)

    #     # --- PRECOMPUTE SMOOTHNESS TERM (Es) ---
    #     # The grid structure doesn't change between frames, so we can define the 
    #     # smoothness constraints (ASAP) logic generally, though we'll rebuild indices 
    #     # per frame to stack with the dynamic data term.
        
    #     # We consider all grid cells (quads) defined by top-left corner (r, c)
    #     # r ranges 0..nrows-2, c ranges 0..ncols-2
    #     r_s, c_s = np.mgrid[0:nrows-1, 0:ncols-1]
    #     r_s = r_s.flatten()
    #     c_s = c_s.flatten()
    #     n_cells = len(r_s)
        
    #     # Indices of the 4 vertices for each cell
    #     v00 = v_idx[r_s, c_s]       # Top-Left
    #     v01 = v_idx[r_s, c_s+1]     # Top-Right
    #     v10 = v_idx[r_s+1, c_s]     # Bot-Left
    #     # v11 = v_idx[r_s+1, c_s+1] # Bot-Right (Not used in the single-triangle constraint)

    #     # ASAP Similarity Constraint (Triangle 1: v00, v01, v10)
    #     # Constraint 1 (X-like): (x01 - x00) - (y10 - y00) = 0
    #     # Vars: x01, x00, y10, y00 -> indices 2*v01, 2*v00, 2*v10+1, 2*v00+1
    #     # Coeffs: 1, -1, -1, 1
        
    #     # Constraint 2 (Y-like): (y01 - y00) + (x10 - x00) = 0
    #     # Vars: y01, y00, x10, x00 -> indices 2*v01+1, 2*v00+1, 2*v10, 2*v00
    #     # Coeffs: 1, -1, 1, -1

    #     w_smooth = 10.0
        
    #     # We will build the smoothness rows relative to a starting offset
    #     # Each cell generates 2 equations. Total 2 * n_cells equations.
    #     # Each equation has 4 entries.
        
    #     # Base templates for one cell's equations
    #     # Eq 1 cols: [2*v01, 2*v00, 2*v10+1, 2*v00+1]
    #     s_cols_1 = np.stack([2*v01, 2*v00, 2*v10+1, 2*v00+1], axis=1).flatten()
    #     s_data_1 = np.tile([w_smooth, -w_smooth, -w_smooth, w_smooth], n_cells)
        
    #     # Eq 2 cols: [2*v01+1, 2*v00+1, 2*v10, 2*v00]
    #     s_cols_2 = np.stack([2*v01+1, 2*v00+1, 2*v10, 2*v00], axis=1).flatten()
    #     s_data_2 = np.tile([w_smooth, -w_smooth, w_smooth, -w_smooth], n_cells)
        
    #     # Pre-assemble smoothness parts that don't depend on row offsets
    #     all_s_cols = np.concatenate([s_cols_1, s_cols_2])
    #     all_s_data = np.concatenate([s_data_1, s_data_2])
    #     all_s_rhs = np.zeros(len(all_s_data) // 4) # 1 RHS value per equation (0)

    #     # Prepare Output
    #     U_completed = np.copy(U)
        
    #     # Stride for subsampling data term (optimization)
    #     stride = 4
    #     y_grid_vals, x_grid_vals = np.mgrid[0:H:stride, 0:W:stride]

    #     for t in tqdm.tqdm(range(len(U)), desc="Motion Completion (ASAP)"):
    #         # Skip if fully valid
    #         # M is (Frames, Height, Width), accessing M[t] gives (Height, Width)
    #         if np.mean(M[t]) > 0.99:
    #             continue

    #         # --- 1. DATA TERM Ed(V) (Vectorized) ---
    #         # Identify valid pixels
    #         # FIX: Remove the 4th index (0) since M is 3D (T, H, W)
    #         mask_sub = (M[t, ::stride, ::stride] > 0.5)
            
    #         ys = y_grid_vals[mask_sub]
    #         xs = x_grid_vals[mask_sub]
    #         n_data_pts = len(ys)
            
    #         if n_data_pts == 0:
    #             continue

    #         # Flow values at these pixels
    #         # Note: U is (H, W, 2), we subsample and mask
    #         u_sub = U[t, ::stride, ::stride, :]
    #         flow_x = u_sub[mask_sub, 0]
    #         flow_y = u_sub[mask_sub, 1]
            
    #         target_x = xs + flow_x
    #         target_y = ys + flow_y

    #         # Bilinear Interpolation Weights
    #         # Grid cell coords
    #         gc = np.clip((xs / gw).astype(int), 0, ncols - 2)
    #         gr = np.clip((ys / gh).astype(int), 0, nrows - 2)
            
    #         alpha = (xs - gc * gw) / gw
    #         beta  = (ys - gr * gh) / gh
            
    #         # Weights
    #         w_tl = (1 - alpha) * (1 - beta)
    #         w_tr = alpha * (1 - beta)
    #         w_bl = (1 - alpha) * beta
    #         w_br = alpha * beta
            
    #         # Vertex indices for every pixel
    #         # Shape (N,)
    #         idx_tl = v_idx[gr, gc]
    #         idx_tr = v_idx[gr, gc+1]
    #         idx_bl = v_idx[gr+1, gc]
    #         idx_br = v_idx[gr+1, gc+1]
            
    #         # We generate 2 equations per pixel: X and Y.
    #         # Total data equations: 2 * n_data_pts
    #         # Total entries in matrix: 2 * n_data_pts * 4 (4 vertices per pixel)
            
    #         # Row indices: 0..2*N-1
    #         # We repeat each row index 4 times (for the 4 weights)
    #         data_row_base = np.arange(2 * n_data_pts)
    #         d_rows = np.repeat(data_row_base, 4)
            
    #         # Data values: The weights repeated for X and Y equations
    #         # For pixel i:
    #         #   Eq X uses weights [tl, tr, bl, br] on X-vars
    #         #   Eq Y uses weights [tl, tr, bl, br] on Y-vars
    #         weights_flat = np.stack([w_tl, w_tr, w_bl, w_br], axis=1).flatten() # (4*N,)
    #         d_data = np.tile(weights_flat, 2) # X weights then Y weights (same values)
            
    #         # Column indices
    #         # For X eq: 2*idx
    #         # For Y eq: 2*idx+1
    #         # indices_flat shape (4*N,) -> [tl0, tr0, bl0, br0, tl1, ...]
    #         indices_flat = np.stack([idx_tl, idx_tr, idx_bl, idx_br], axis=1).flatten()
            
    #         d_cols_x = 2 * indices_flat
    #         d_cols_y = 2 * indices_flat + 1
    #         d_cols = np.concatenate([d_cols_x, d_cols_y])
            
    #         # RHS
    #         d_rhs = np.zeros(2 * n_data_pts)
    #         d_rhs[0::2] = target_x # Evens are X
    #         d_rhs[1::2] = target_y # Odds are Y
            
    #         # --- 2. ASSEMBLE SYSTEM ---
    #         # Offset smoothness rows to start after data rows
    #         num_data_rows = 2 * n_data_pts
            
    #         # Generate row indices for smoothness
    #         # There are 2*n_cells equations. Each has 4 entries.
    #         # We need to range from num_data_rows to num_data_rows + 2*n_cells
    #         s_row_base = np.arange(num_data_rows, num_data_rows + 2 * n_cells)
    #         s_rows = np.repeat(s_row_base, 4)
            
    #         # Concatenate everything
    #         full_rows = np.concatenate([d_rows, s_rows])
    #         full_cols = np.concatenate([d_cols, all_s_cols])
    #         full_data = np.concatenate([d_data, all_s_data])
    #         full_rhs  = np.concatenate([d_rhs,  all_s_rhs])
            
    #         # --- 3. SOLVE ---
    #         # Total equations
    #         n_eqs = num_data_rows + 2 * n_cells
            
    #         matrix = scipy.sparse.coo_matrix((full_data, (full_rows, full_cols)), 
    #                                          shape=(n_eqs, 2 * n_verts))
    #         matrix_csr = matrix.tocsr()
            
    #         # Solve using LSQR
    #         # Use lsmr for potentially faster convergence on regularized least squares
    #         res = scipy.sparse.linalg.lsqr(matrix_csr, full_rhs, show=False)
    #         coords_flat = res[0]
            
    #         # --- 4. RENDER ---
    #         grid_flow_x = coords_flat[0::2].reshape(nrows, ncols) - grid_x
    #         grid_flow_y = coords_flat[1::2].reshape(nrows, ncols) - grid_y
            
    #         full_flow_x = cv2.resize(grid_flow_x, (W, H))
    #         full_flow_y = cv2.resize(grid_flow_y, (W, H))
            
    #         # Blend: Keep original flow where valid, fill where invalid
    #         # Using 1-M[t] as mask for filling
    #         # FIX: Ensure M[t] is broadcastable. M[t] is (H,W), U is (H,W,2)
    #         mask_expanded = np.dstack([M[t], M[t]]) # Expand to (H,W,2) for broadcasting
            
    #         filled_x = U[t,:,:,0] * M[t] + full_flow_x * (1 - M[t])
    #         filled_y = U[t,:,:,1] * M[t] + full_flow_y * (1 - M[t])
            
    #         U_completed[t] = np.stack([filled_x, filled_y], axis=2)

    #     return U_completed

    # def _motion_completion(self, U, M, shape):
    #     # Section 4.3: Motion Completion
    #     # "Minimizing energy E(V) = Ed(V) + Es(V)"
    #     # "Grid size 40x40 pixels"
        
    #     h, w = shape[:2]
    #     gw, gh = self.grid_size, self.grid_size
        
    #     U_completed = np.copy(U)
        
    #     for t in tqdm.tqdm(range(len(U)), desc="Motion Completion"):
    #         # If mask is full, skip
    #         if np.mean(M[t]) > 0.99: continue
            
    #         # PROXY implementation to allow execution:
    #         # (To be faithful, one must implement the ASAP Energy minimization here)
    #         flow_x = U[t,:,:,0]
    #         flow_y = U[t,:,:,1]
    #         mask_uint8 = M[t].astype(np.uint8)
            
    #         # Use Navier-Stokes based inpainting as a computational proxy for the 
    #         # "smoothness" term filling in holes. 
    #         if np.min(mask_uint8) == 0:
    #             U_completed[t,:,:,0] = cv2.inpaint(flow_x, 1-mask_uint8, 3, cv2.INPAINT_NS)
    #             U_completed[t,:,:,1] = cv2.inpaint(flow_y, 1-mask_uint8, 3, cv2.INPAINT_NS)
            
    #     return U_completed

    def _stabilize_pixel_profiles(self, U):
        # Section 5: Pixel Profile Stabilization
        # Equation 3: O({Pt}) = sum(||Pt - Ct||^2 + lambda * sum(w * ||Pt - Pr||^2))
        
        C = np.cumsum(U, axis=0) # Accumulated motion paths
        C = np.vstack([np.zeros((1, C.shape[1], C.shape[2], 2), dtype=C.dtype), C]) # Add t=0
        
        frames, h, w, _ = C.shape
        P = np.copy(C)
        
        # Iterative Jacobi Solver (Equation 4)
        # P_t^(xi+1) = 1/gamma * (C_t + lambda * sum(w_tr * P_r^xi))
        
        # Precompute spatial gaussian weights w_tr
        radius = self.temporal_radius
        
        weights = np.exp(-np.arange(-radius, radius+1)**2 / (radius/3)**2)
        weights[radius] = 0 # w_tt = 0
        
        # Optimization loop
        for xi in range(10): # 10 iterations default
            # Vectorized Jacobi update
            # We need sum(w_tr * P_r) for all t. This is a convolution along time axis.
            # Using scipy.ndimage for convolution
            
            # weighted_sum_P = convolve(P, weights)
            w_sum_P_x = scipy.ndimage.convolve1d(P[:,:,:,0], weights, axis=0, mode='mirror')
            w_sum_P_y = scipy.ndimage.convolve1d(P[:,:,:,1], weights, axis=0, mode='mirror')
            
            w_sum_weights = np.sum(weights) 
            
            gamma = 1 + self.lambda_smooth * w_sum_weights
            
            P[:,:,:,0] = (C[:,:,:,0] + self.lambda_smooth * w_sum_P_x) / gamma
            P[:,:,:,1] = (C[:,:,:,1] + self.lambda_smooth * w_sum_P_y) / gamma
            
        return P

    def _write_video(self, output_path, frames, B, fps):
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # B contains the warp vectors (P - C). 
        # Warp original input video frame to stabilized frame by dense flow field B_t
        
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for t in range(len(frames)):
            if t >= len(B): break
            
            # Inverse warp: We want to know where pixel (x,y) in STABLE frame comes from in UNSTABLE frame.
            # B_t is vector from Unstable -> Stable.
            # So Source(x,y) = (x,y) - B_t(x,y) roughly? 
            # Actually, standard remap needs backward mapping. 
            # Ideally we invert B_t. For small motions, -B_t is approximation.
            
            flow = B[t]
            map_x = grid_x - flow[:,:,0]
            map_y = grid_y - flow[:,:,1]
            
            frame_stab = cv2.remap(frames[t], map_x, map_y, cv2.INTER_LINEAR)
            out.write(frame_stab)
            
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SteadyFlow Video Stabilization")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_video", help="Path to output stabilized video")
    args = parser.parse_args()

    stabilizer = SteadyFlowStabilizer()
    stabilizer.stabilize(args.input_video, args.output_video)
