# Basic, step bystep implementation of the paper

import cv2
import numpy as np
import sys
import os
import shutil
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix, linalg

# ==========================================
# 1. Motion Completion (Sparse Solver)
# ==========================================
class MotionCompleter:
    """
    Handles the 'Motion Completion' step (Section 4.3) using sparse linear solvers.
    Fills in missing flow regions (M=0) using ASAP warping.
    """
    def __init__(self, width, height, grid_size=40, lambda_smooth=1.0):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.lambda_smooth = lambda_smooth

        self.cols = int(np.ceil(width / grid_size)) + 1
        self.rows = int(np.ceil(height / grid_size)) + 1
        self.num_vars = self.rows * self.cols * 2
        self.triangles = self._generate_triangles()

    def _generate_triangles(self):
        triangles = []
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                v_tl = r * self.cols + c
                v_tr = r * self.cols + (c + 1)
                v_bl = (r + 1) * self.cols + c
                v_br = (r + 1) * self.cols + (c + 1)
                triangles.append((v_tl, v_tr, v_bl))
                triangles.append((v_tr, v_br, v_bl))
        return triangles

    def _get_bilinear_weights(self, y_px, x_px):
        gx = x_px / self.grid_size
        gy = y_px / self.grid_size
        
        ix = np.clip(np.floor(gx).astype(int), 0, self.cols - 2)
        iy = np.clip(np.floor(gy).astype(int), 0, self.rows - 2)

        fx = gx - ix
        fy = gy - iy

        v_tl = iy * self.cols + ix
        v_tr = iy * self.cols + (ix + 1)
        v_bl = (iy + 1) * self.cols + ix
        v_br = (iy + 1) * self.cols + (ix + 1)

        w_tl = (1 - fx) * (1 - fy)
        w_tr = fx * (1 - fy)
        w_bl = (1 - fx) * fy
        w_br = fx * fy

        return np.stack([v_tl, v_tr, v_bl, v_br], axis=1), \
               np.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    def solve(self, flow_field, mask):
        # Subsample for performance
        stride = 8 
        y_grid, x_grid = np.mgrid[0:self.height:stride, 0:self.width:stride]
        
        valid_mask = mask[0:self.height:stride, 0:self.width:stride] > 0.5
        
        pts_y = y_grid[valid_mask]
        pts_x = x_grid[valid_mask]
        
        u_valid = flow_field[0:self.height:stride, 0:self.width:stride, 0][valid_mask]
        v_valid = flow_field[0:self.height:stride, 0:self.width:stride, 1][valid_mask]
        
        target_x = pts_x + u_valid
        target_y = pts_y + v_valid
        
        N_data = len(pts_x)
        if N_data < 4: return flow_field

        v_indices, weights = self._get_bilinear_weights(pts_y, pts_x)
        
        rows, cols, data = [], [], []
        b_vec = []
        eqn_idx = 0

        # Data Term
        for i in range(N_data):
            # X
            rows.extend([eqn_idx]*4)
            cols.extend(v_indices[i] * 2)
            data.extend(weights[i])
            b_vec.append(target_x[i])
            eqn_idx += 1
            # Y
            rows.extend([eqn_idx]*4)
            cols.extend(v_indices[i] * 2 + 1)
            data.extend(weights[i])
            b_vec.append(target_y[i])
            eqn_idx += 1

        # Smoothness Term (ASAP)
        w_smooth = self.lambda_smooth * 10
        for v_tl, v_tr, v_bl in self.triangles:
            rows.extend([eqn_idx]*4)
            cols.extend([v_tr*2, v_tl*2, v_bl*2+1, v_tl*2+1])
            data.extend([w_smooth, -w_smooth, -w_smooth, w_smooth])
            b_vec.append(0)
            eqn_idx += 1
            
            rows.extend([eqn_idx]*4)
            cols.extend([v_tr*2+1, v_tl*2+1, v_bl*2, v_tl*2])
            data.extend([w_smooth, -w_smooth, w_smooth, -w_smooth])
            b_vec.append(0)
            eqn_idx += 1

        A = coo_matrix((data, (rows, cols)), shape=(eqn_idx, self.num_vars))
        b = np.array(b_vec)
        
        res = linalg.lsqr(A, b, show=False)
        sol = res[0]
        
        grid_pos = sol.reshape((self.rows, self.cols, 2))
        
        # Warp back
        # We need to compute flow displacement
        orig_y, orig_x = np.mgrid[0:self.rows, 0:self.cols]
        orig_y *= self.grid_size
        orig_x *= self.grid_size
        
        grid_flow = grid_pos - np.dstack((orig_x, orig_y))
        
        map_x = cv2.resize(grid_flow[:,:,0], (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(grid_flow[:,:,1], (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        
        return np.dstack((map_x, map_y)).astype(np.float32)


# ==========================================
# 2. Pixel Profile Stabilization
# ==========================================
class PixelProfileStabilizer:
    """
    Implements Section 5: Smoothing pixel profiles.
    Used in both the final step AND inside the iterative refinement loop.
    """
    def __init__(self, tau=20, lambda_smooth=50.0, iterations=5):
        self.tau = tau
        self.lambda_smooth = lambda_smooth
        self.iterations = iterations

    def _compute_adaptive_windows(self, C):
        """
        Implements Adaptive Window Selection (Section 5.1).
        Intersection of windows for all pixels: max deviation must be < tau.
        """
        T, H, W, _ = C.shape
        window_starts = np.zeros(T, dtype=int)
        window_ends = np.zeros(T, dtype=int)

        # Optimization: Downsample C for window estimation
        stride = 8
        C_small = C[:, ::stride, ::stride, :]
        
        for t in range(T):
            # Backwards
            l = t
            while l > 0:
                # "global smooth window... intersection of the windows at all pixels" [cite: 1160]
                # This means we check the MAX deviation across the whole frame.
                diff = np.abs(C_small[t] - C_small[l-1])
                if np.max(diff) > self.tau:
                    break
                l -= 1
            window_starts[t] = l
            
            # Forwards
            r = t
            while r < T - 1:
                diff = np.abs(C_small[t] - C_small[r+1])
                if np.max(diff) > self.tau:
                    break
                r += 1
            window_ends[t] = r
            
        return window_starts, window_ends

    def stabilize(self, C):
        """
        Stabilizes the accumulated motion C using Jacobi iterations.
        Returns P (Stabilized Accumulated Motion).
        """
        T, H, W, _ = C.shape
        
        # 1. Adaptive Windows
        starts, ends = self._compute_adaptive_windows(C)
        
        # 2. Initialize P
        P = np.copy(C)
        
        # 3. Jacobi Iterations
        for _ in range(self.iterations):
            P_new = np.zeros_like(P)
            
            # Iterate per frame
            for t in range(T):
                start, end = starts[t], ends[t]
                
                # Sigma for Gaussian weight
                # "spatial Gaussian function... exp(-||r-t||^2 / (Omega_t/3)^2)" [cite: 1142]
                window_len = end - start
                if window_len < 1: window_len = 1
                sigma = window_len / 3.0
                
                numerator = np.zeros((H, W, 2), dtype=np.float32)
                denom = 0.0
                
                # Sum over window
                for r in range(start, end + 1):
                    if r == t: continue
                    weight = np.exp(-((r - t)**2) / (sigma**2))
                    numerator += weight * P[r]
                    denom += weight
                
                # Eq 4: P = 1/gamma * (C + lambda * sum(w*P)) [cite: 1146]
                gamma = 1 + self.lambda_smooth * denom
                P_new[t] = (C[t] + self.lambda_smooth * numerator) / gamma
            
            P = P_new
            
        return P


# ==========================================
# 3. Discontinuity Analysis
# ==========================================
class DiscontinuityAnalyzer:
    def spatial_check(self, flow):
        """Threshold gradient magnitude [cite: 3005]"""
        u, v = flow[:,:,0], flow[:,:,1]
        gy, gx = np.gradient(u)
        gy2, gx2 = np.gradient(v)
        grad = np.sqrt(gx**2 + gy**2 + gx2**2 + gy2**2)
        return np.where(grad > 0.1, 0.0, 1.0)

    def temporal_check(self, stabilized_accum, current_accum, iteration):
        """
        Checks difference between current trajectory and stabilized trajectory.
        Uses adaptive threshold (1 + 20^(1/n)) * epsilon [cite: 1130]
        """
        diff = np.linalg.norm(stabilized_accum - current_accum, axis=2) # (H, W)
        thresh = (1 + 20.0**(1/iteration)) * 0.2
        return np.where(diff > thresh, 0.0, 1.0)


# ==========================================
# 4. Main Pipeline (SteadyFlowAutomator)
# ==========================================
class SteadyFlowAutomator:
    def __init__(self, input_path, output_path, temp_dir='steady_temp'):
        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.completer = MotionCompleter(self.width, self.height)
        self.analyzer = DiscontinuityAnalyzer()
        # Stabilizer for the loop (lighter settings)
        self.loop_stabilizer = PixelProfileStabilizer(tau=20, lambda_smooth=50.0, iterations=5)
        # Stabilizer for final pass (higher quality)
        self.final_stabilizer = PixelProfileStabilizer(tau=20, lambda_smooth=100.0, iterations=15)

    def _get_memmap(self, name, shape, dtype=np.float32):
        path = os.path.join(self.temp_dir, f"{name}.dat")
        mode = 'w+' if not os.path.exists(path) else 'r+'
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    def run(self):
        print(f"Processing {self.num_frames} frames ({self.width}x{self.height})...")
        
        # 1. Initialization
        self._step_initialization()
        
        # 2. Iterative Refinement
        self._step_iterative_refinement(iterations=3)
        
        # 3. Final Stabilization
        self._step_final_stabilization()
        
        # 4. Rendering
        self._step_rendering()
        
        self.cap.release()
        try: shutil.rmtree(self.temp_dir)
        except: pass
        print("Done.")

    def _step_initialization(self):
        print("Step 1: Robust Initialization (Backward Flow)...")
        flows = self._get_memmap('flows', (self.num_frames, self.height, self.width, 2))
        
        prev_frame = None
        for i in range(self.num_frames):
            ret, curr_frame = self.cap.read()
            if not ret: break
            
            if i == 0:
                prev_frame = curr_frame
                continue # Flow at t=0 is 0
                
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # KLT + Homography (Curr -> Prev)
            p0 = cv2.goodFeaturesToTrack(curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            H = np.eye(3, dtype=np.float32)
            if p0 is not None:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p0, None)
                good_curr = p0[status == 1]
                good_prev = p1[status == 1]
                if len(good_curr) >= 4:
                    H, _ = cv2.findHomography(good_curr, good_prev, cv2.RANSAC, 5.0)
            
            # Align: Warp Curr to Prev geometry
            curr_aligned = cv2.warpPerspective(curr_gray, H, (self.width, self.height))
            
            # Residual Flow (Curr_aligned -> Prev)
            flow_res = cv2.calcOpticalFlowFarneback(curr_aligned, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Global Flow from H
            y, x = np.mgrid[0:self.height, 0:self.width]
            h = H.flatten()
            denom = h[6]*x + h[7]*y + h[8]
            denom[np.abs(denom) < 1e-5] = 1e-5
            gx = (h[0]*x + h[1]*y + h[2]) / denom - x
            gy = (h[3]*x + h[4]*y + h[5]) / denom - y
            flow_global = np.dstack((gx, gy))
            
            flows[i] = flow_res + flow_global
            prev_frame = curr_frame
            sys.stdout.write(f"\r  Init Frame {i+1}")
        flows.flush()
        print()

    def _step_iterative_refinement(self, iterations):
        print("Step 2: Iterative Refinement...")
        flows = self._get_memmap('flows', (self.num_frames, self.height, self.width, 2))
        
        # Buffer for masks
        masks = self._get_memmap('masks', (self.num_frames, self.height, self.width), dtype=np.float32)
        
        # Buffer for Accumulated Motion C (re-used)
        C = self._get_memmap('accum_C', (self.num_frames, self.height, self.width, 2))
        
        for it in range(1, iterations + 1):
            print(f"  Iteration {it}/{iterations}")
            
            # --- 1. Identify Spatial Discontinuities ---
            # Paper says: "In practice, we obtain an initial outlier mask Mt... only by spatial analysis"
            # then refine. In loop, we re-calc spatial on refined flow.
            for i in range(self.num_frames):
                masks[i] = self.analyzer.spatial_check(flows[i])
            
            # --- 2. Temporal Analysis (if it > 1) ---
            # To do temporal analysis, we need a stabilized reference.
            # "The stabilized SteadyFlow is then used to further refine Mt by temporal analysis" 
            if it > 1:
                # Calc C from current flows
                np.cumsum(flows, axis=0, out=C)
                
                # Stabilize (Pixel Profile Stabilization inside loop)
                print("    Stabilizing for Temporal Analysis...")
                P = self.loop_stabilizer.stabilize(C)
                
                # Update masks
                for i in range(self.num_frames):
                    t_mask = self.analyzer.temporal_check(P[i], C[i], it)
                    masks[i] *= t_mask # Intersection
            
            # --- 3. Motion Completion ---
            print("    Completing Motion...")
            for i in range(self.num_frames):
                if np.mean(masks[i]) < 0.99:
                    completed = self.completer.solve(flows[i], masks[i])
                    flows[i] = completed
                sys.stdout.write(f"\r    Frame {i+1}")
            flows.flush()
            print()

    def _step_final_stabilization(self):
        print("Step 3: Final Pixel Profile Stabilization...")
        # Load refined flows
        flows = self._get_memmap('flows', (self.num_frames, self.height, self.width, 2))
        
        # Compute Accumulation C
        C = self._get_memmap('accum_C', (self.num_frames, self.height, self.width, 2))
        np.cumsum(flows, axis=0, out=C)
        
        # Final Stabilization
        # P stores the smooth paths
        P = self._get_memmap('final_P', (self.num_frames, self.height, self.width, 2))
        
        # We cannot load full P into RAM for solve, but PixelProfileStabilizer
        # currently expects in-memory arrays. 
        # For large videos, we must process spatially in chunks or modify the stabilizer.
        # Given the complexity, we assume here that (T,H,W,2) fits in RAM *or* swap.
        # If not, we would slice C spatially.
        
        print("  Running Jacobi Optimization...")
        P_solved = self.final_stabilizer.stabilize(C)
        P[:] = P_solved[:]
        P.flush()

    def _step_rendering(self):
        print("Step 4: Rendering...")
        C = self._get_memmap('accum_C', (self.num_frames, self.height, self.width, 2))
        P = self._get_memmap('final_P', (self.num_frames, self.height, self.width, 2))
        
        # B = P - C (Displacement from Input to Smooth)
        # To get pixel at P (smooth target), we look at C (input).
        # Inverse warp: map = grid + (C - P)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width]
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)
        
        for i in range(self.num_frames):
            ret, frame = self.cap.read()
            if not ret: break
            
            # diff = C - P
            diff = C[i] - P[i]
            
            map_x = grid_x + diff[:,:,0]
            map_y = grid_y + diff[:,:,1]
            
            stabilized = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out.write(stabilized)
            sys.stdout.write(f"\r  Render {i+1}")
            
        out.release()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python steadyflow.py <input.mp4> <output.mp4>")
    else:
        automator = SteadyFlowAutomator(sys.argv[1], sys.argv[2])
        automator.run()
