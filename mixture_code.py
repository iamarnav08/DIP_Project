import cv2
import numpy as np
import sys
import tqdm
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. MeshFlow-based Motion Generator
#    (Replaces brittle Farneback + ASAP Warping)
# ==========================================
class MeshFlowGenerator:
    """
    Generates a 'SteadyFlow' (spatially smooth dense flow) using the 
    robust sparse-to-dense logic from the MeshFlow paper[cite: 400].
    """
    def __init__(self, width, height, mesh_rows=16, mesh_cols=16):
        self.width = width
        self.height = height
        self.rows = mesh_rows
        self.cols = mesh_cols
        self.feature_detector = cv2.FastFeatureDetector_create()
        
        # Grid definition
        self.x_grid = np.linspace(0, width, mesh_cols + 1)
        self.y_grid = np.linspace(0, height, mesh_rows + 1)

    def get_smooth_dense_flow(self, prev_gray, curr_gray):
        # 1. Detect Features (FAST) [cite: 496]
        p0 = self.feature_detector.detect(prev_gray)
        if len(p0) < 10: 
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        p0_pts = np.float32([k.pt for k in p0]).reshape(-1, 1, 2)
        
        # 2. Track Features (KLT) [cite: 496]
        p1_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0_pts, None)
        status = status.reshape(-1)
        
        good_p0 = p0_pts[status == 1].reshape(-1, 2)
        good_p1 = p1_pts[status == 1].reshape(-1, 2)
        
        # 3. Estimate Global Homography (for fallback)
        if len(good_p0) < 4:
            return np.zeros((self.height, self.width, 2), dtype=np.float32)
        H_global, _ = cv2.findHomography(good_p0, good_p1, cv2.RANSAC)
        
        # 4. Propagate Motion to Mesh Vertices [cite: 435]
        # Initialize mesh motion with global homography to handle textureless regions
        mesh_motion_x = np.zeros((self.rows + 1, self.cols + 1), dtype=np.float32)
        mesh_motion_y = np.zeros((self.rows + 1, self.cols + 1), dtype=np.float32)
        mesh_counts = np.zeros((self.rows + 1, self.cols + 1), dtype=np.float32)

        # Pre-fill with global motion
        grid_y, grid_x = np.meshgrid(self.y_grid, self.x_grid, indexing='ij')
        ones = np.ones_like(grid_x)
        coords = np.stack([grid_x.flatten(), grid_y.flatten(), ones.flatten()])
        warped_coords = H_global @ coords
        warped_coords /= warped_coords[2, :]
        global_dx = warped_coords[0, :] - grid_x.flatten()
        global_dy = warped_coords[1, :] - grid_y.flatten()
        
        mesh_motion_x = global_dx.reshape(self.rows + 1, self.cols + 1)
        mesh_motion_y = global_dy.reshape(self.rows + 1, self.cols + 1)

        # Accumulate local motion from features (The "Ellipse" propagation) 
        motion = good_p1 - good_p0
        
        for (pt, mot) in zip(good_p0, motion):
            # Find closest grid vertex
            c = int(round(pt[0] / self.width * self.cols))
            r = int(round(pt[1] / self.height * self.rows))
            
            # Simple 3x3 window propagation (simplified ellipse)
            r_min, r_max = max(0, r-1), min(self.rows, r+1)
            c_min, c_max = max(0, c-1), min(self.cols, c+1)
            
            mesh_motion_x[r_min:r_max+1, c_min:c_max+1] += mot[0]
            mesh_motion_y[r_min:r_max+1, c_min:c_max+1] += mot[1]
            mesh_counts[r_min:r_max+1, c_min:c_max+1] += 1

        # Average the accumulated motions
        mask = mesh_counts > 0
        # If we have local features, mix them with global. If not, keep global.
        # (Simplified for brevity: in production, you'd weight this)
        mesh_motion_x[mask] = mesh_motion_x[mask] / mesh_counts[mask]
        mesh_motion_y[mask] = mesh_motion_y[mask] / mesh_counts[mask]

        # 5. Spatial Smoothing (Median Filter f2) [cite: 515]
        # This replaces SteadyFlow's complex ASAP Motion Completion
        mesh_motion_x = cv2.medianBlur(mesh_motion_x.astype(np.float32), 3)
        mesh_motion_y = cv2.medianBlur(mesh_motion_y.astype(np.float32), 3)

        # 6. Bilinear Upsampling to Dense Flow (The "SteadyFlow" Representation)
        # SteadyFlow requires dense flow at every pixel. We interpolate the mesh.
        map_x = cv2.resize(mesh_motion_x, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(mesh_motion_y, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        return np.dstack([map_x, map_y]), H_global

# ==========================================
# 2. Pixel Profile Stabilization (SteadyFlow Logic)
#    (Faithful to SteadyFlow Section 5)
# ==========================================
class PixelProfileStabilizer:
    def __init__(self, width, height, lambda_smooth=10.0):
        self.width = width
        self.height = height
        self.lambda_smooth = lambda_smooth # Base lambda
        self.radius = 100 # Omega_t [cite: 255]

    def compute_adaptive_weights(self, homographies):
        # Implementation of PAPS from MeshFlow Section 4.2 
        # "PAPS is proposed to shift the method online... but we use prediction"
        T = len(homographies)
        lambdas = np.zeros(T)
        
        for t in range(T):
            H = homographies[t]
            # Translational element
            trans = np.sqrt((H[0,2]/self.width)**2 + (H[1,2]/self.height)**2)
            # Affine Component (eigenvalue ratio)
            Affine = H[:2, :2]
            # Fix: Handle identity/small affine cases
            if np.abs(np.linalg.det(Affine)) < 1e-5:
                ratio = 1.0
            else:
                eigs = np.sort(np.abs(np.linalg.eigvals(Affine)))
                ratio = eigs[-2] / (eigs[-1] + 1e-8)
            
            # Linear models from MeshFlow 
            lam_1 = -1.93 * trans + 0.95
            lam_2 = 5.83 * ratio + 4.88 # Note: Paper says ratio of largest 2 eigs
            
            # Combined weight
            lam = max(min(lam_1, lam_2), 0.0) # Lower bound 0
            lambdas[t] = lam * self.lambda_smooth
            
        return lambdas

    def stabilize(self, C, homographies):
        """
        Solves Eq. 3 from SteadyFlow paper: 
        Minimize ||P - C||^2 + lambda * sum(w * ||P_t - P_r||^2)
        """
        T, H, W, _ = C.shape
        lambdas = self.compute_adaptive_weights(homographies)
        P = np.copy(C)
        
        # Pre-compute Gaussian weights window [cite: 238]
        sigma = self.radius / 3.0
        k_size = int(self.radius * 2 + 1)
        # Standard Gaussian Window
        window = cv2.getGaussianKernel(k_size, sigma).reshape(-1)
        window[self.radius] = 0 # w_{t,t} = 0
        
        # Jacobi Iteration 
        # "We solve it iteratively by a Jacobi-based solver"
        iterations = 10
        print("Optimizing Pixel Profiles...")
        
        for it in range(iterations): 
            # Separable convolution for speed (equivalent to sum over window)
            # We treat the time axis as the dimension to filter
            
            # P is (T, H, W, 2). Move Time to last for applying filter? 
            # Actually, gaussian_filter1d works on the specified axis.
            
            Px = P[:,:,:,0]
            Py = P[:,:,:,1]
            
            # Calculate weighted sum of neighbors (The second term in Eq 4)
            # sum_{r in Omega} w_{t,r} * P_r
            Sum_Px = gaussian_filter1d(Px, sigma=sigma, axis=0, mode='nearest', truncate=3.0) 
            Sum_Py = gaussian_filter1d(Py, sigma=sigma, axis=0, mode='nearest', truncate=3.0)
            
            # Note: gaussian_filter1d includes the center element. We must subtract it 
            # to strictly follow the "neighbors only" Jacobi formulation, 
            # OR we simply use the filter result as the "target" smooth path.
            # SteadyFlow Eq 4: P_new = (C + lambda * sum(w*P_old)) / (1 + lambda * sum(w))
            
            # Re-normalization factor for the window
            w_sum = np.sum(window) # Approx 1.0 usually
            
            # Update P 
            # Since lambda changes per frame, we iterate by frame
            for t in range(T):
                lam = lambdas[t]
                gamma = 1.0 + lam * w_sum
                
                # Update rule
                P[t] = (C[t] + lam * w_sum * np.dstack((Sum_Px[t], Sum_Py[t]))) / gamma
                
        return P

# ==========================================
# Main Process
# ==========================================
class SteadyFlow_Refined:
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_path = output_path
        
        # Downscale for processing speed (MeshFlow works on low res grids)
        # SteadyFlow also mentions processing flow on downsampled images.
        self.proc_w = 320
        self.proc_h = int(self.height * (320 / self.width))
        
        # Use MeshFlow Generator logic
        self.flow_generator = MeshFlowGenerator(self.proc_w, self.proc_h)
        self.stabilizer = PixelProfileStabilizer(self.proc_w, self.proc_h, lambda_smooth=10.0)

    def process(self):
        print(f"Processing {self.num_frames} frames...")
        
        flows = []
        homographies = []
        
        ret, prev = self.cap.read()
        if not ret: return
        prev_gray = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (self.proc_w, self.proc_h))
        
        # Initial dummy flow/homography
        flows.append(np.zeros((self.proc_h, self.proc_w, 2), dtype=np.float32))
        homographies.append(np.eye(3, dtype=np.float32))
        
        # 1. Flow Calculation (The "SteadyFlow Estimation" phase)
        for _ in tqdm.tqdm(range(self.num_frames - 1), desc="Generating SteadyFlow"):
            ret, curr = self.cap.read()
            if not ret: break
            curr_gray = cv2.resize(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (self.proc_w, self.proc_h))

            # 1. Get BOTH flow and Homography from the generator
            dense_flow, H_global = self.flow_generator.get_smooth_dense_flow(prev_gray, curr_gray)

            flows.append(dense_flow)

            # 2. Use the returned H directly (No need to re-calculate!)
            homographies.append(H_global)

            prev_gray = curr_gray
            
        flows = np.array(flows)
        homographies = np.array(homographies)

        # 2. Accumulate Motion (C_t in Eq 3) [cite: 236]
        # "C_t is the field of accumulated motion vectors"
        print("Accumulating Motion...")
        C = np.cumsum(flows, axis=0)

        # 3. Stabilization (Smoothing Pixel Profiles) [cite: 231]
        P = self.stabilizer.stabilize(C, homographies)
        
        # 4. Rendering [cite: 244]
        # "Warp... by a dense flow field B_t = P_t - C_t"
        print("Rendering...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        
        for i in tqdm.tqdm(range(self.num_frames), desc="Warping"):
            ret, frame = self.cap.read()
            if not ret: break
            
            if i < len(P):
                # Calculate Update Field B_t [cite: 244]
                # P and C are accumulated, so the difference is the absolute warping needed
                update = P[i] - C[i]
                
                # Upscale the update field to full resolution
                u_x = cv2.resize(update[:,:,0], (self.width, self.height))
                u_y = cv2.resize(update[:,:,1], (self.width, self.height))
                
                # Apply Inverse Warp (Standard Stabilization)
                # To stabilize, we pull pixels from where they *shook* to (C) 
                # back to where they *should be* (P). 
                # map = grid - displacement
                map_x = (grid_x - u_x).astype(np.float32)
                map_y = (grid_y - u_y).astype(np.float32)
                
                stab = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                writer.write(stab)
            else:
                writer.write(frame)
                
        writer.release()
        self.cap.release()
        print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python steadyflow_refined.py input.mp4 output.mp4")
    else:
        sf = SteadyFlow_Refined(sys.argv[1], sys.argv[2])
        sf.process()
