import cv2
import numpy as np
import sys
import tqdm
import pyflow 
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. Optical Flow Estimation (SteadyFlow Sec 4.1)
# ==========================================
class CeLiuFlowEstimator:
    """
    Implements SteadyFlow Initialization[cite: 3354]:
    1. Estimate Global Homography (H)
    2. Warp Current Frame -> Residual Motion Calculation (using Ce Liu's method)
    3. Combine Global + Residual
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.detector = cv2.FastFeatureDetector_create()
        
        # Ce Liu Parameters (defaults from paper/thesis)
        self.alpha = 0.012
        self.ratio = 0.75
        self.minWidth = 20
        self.nOuterFPIterations = 7
        self.nInnerFPIterations = 1
        self.nSORIterations = 30
        self.colType = 1 

    def compute_flow(self, prev_img, curr_img):
        # A. Estimate Global Homography (KLT + RANSAC)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        
        p0 = self.detector.detect(prev_gray)
        H = np.eye(3, dtype=np.float32)
        
        if len(p0) >= 4:
            p0_pts = np.float32([k.pt for k in p0]).reshape(-1, 1, 2)
            p1_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0_pts, None)
            good_p0 = p0_pts[st == 1]
            good_p1 = p1_pts[st == 1]
            if len(good_p0) >= 4:
                H_est, _ = cv2.findHomography(good_p0, good_p1, cv2.RANSAC)
                if H_est is not None: H = H_est.astype(np.float32)

        # B. Pre-warp current image to align with previous
        # "We align them accordingly" [cite: 3356]
        curr_warped = cv2.warpPerspective(curr_img, np.linalg.inv(H), (self.width, self.height))

        # C. Compute Residual Flow using Ce Liu's algorithm
        # "Apply optical flow... to compute residual motion field" [cite: 3357]
        im1 = prev_img.astype(float) / 255.0
        im2 = curr_warped.astype(float) / 255.0
        im1 = np.ascontiguousarray(im1)
        im2 = np.ascontiguousarray(im2)
        
        u, v, _ = pyflow.coarse2fine_flow(
            im1, im2, self.alpha, self.ratio, self.minWidth, 
            self.nOuterFPIterations, self.nInnerFPIterations, 
            self.nSORIterations, self.colType
        )
        residual_flow = np.dstack((u, v))

        # D. Compose Global + Residual
        # Total = (H(x) - x) + Residual
        y_grid, x_grid = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        ones = np.ones_like(x_grid)
        coords = np.stack([x_grid.flatten(), y_grid.flatten(), ones.flatten()])
        
        new_coords = H @ coords
        new_coords /= (new_coords[2, :] + 1e-8)
        
        global_flow_x = new_coords[0, :].reshape(self.height, self.width) - x_grid
        global_flow_y = new_coords[1, :].reshape(self.height, self.width) - y_grid
        
        total_flow = np.dstack([global_flow_x, global_flow_y]) + residual_flow
        
        return total_flow

# ==========================================
# 2. Motion Completion (SteadyFlow Sec 4.3)
# ==========================================
class SparseMotionCompleter:
    def __init__(self, width, height, grid_size=40):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = int(np.ceil(width / grid_size))
        self.rows = int(np.ceil(height / grid_size))
        self.num_vertices = (self.rows + 1) * (self.cols + 1)

    def fill_holes(self, flow, outlier_mask):
        vertex_flow_x = np.zeros(self.num_vertices)
        vertex_flow_y = np.zeros(self.num_vertices)
        
        grid_mask = cv2.resize((~outlier_mask).astype(np.float32), (self.cols + 1, self.rows + 1))
        grid_flow_x = cv2.resize(flow[:,:,0], (self.cols + 1, self.rows + 1))
        grid_flow_y = cv2.resize(flow[:,:,1], (self.cols + 1, self.rows + 1))
        
        vertex_is_constrained = grid_mask.flatten() > 0.1
        vertex_flow_x = grid_flow_x.flatten()
        vertex_flow_y = grid_flow_y.flatten()

        A = lil_matrix((self.num_vertices, self.num_vertices))
        b_x = np.zeros(self.num_vertices)
        b_y = np.zeros(self.num_vertices)
        
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                idx = r * (self.cols + 1) + c
                if vertex_is_constrained[idx]:
                    A[idx, idx] = 1
                    b_x[idx] = vertex_flow_x[idx]
                    b_y[idx] = vertex_flow_y[idx]
                else:
                    neighbors = []
                    if r > 0: neighbors.append(idx - (self.cols + 1))
                    if r < self.rows: neighbors.append(idx + (self.cols + 1))
                    if c > 0: neighbors.append(idx - 1)
                    if c < self.cols: neighbors.append(idx + 1)
                    A[idx, idx] = len(neighbors)
                    for n_idx in neighbors: A[idx, n_idx] = -1

        solved_x = spsolve(A.tocsr(), b_x)
        solved_y = spsolve(A.tocsr(), b_y)
        
        grid_filled_x = solved_x.reshape(self.rows + 1, self.cols + 1)
        grid_filled_y = solved_y.reshape(self.rows + 1, self.cols + 1)
        
        dense_x = cv2.resize(grid_filled_x, (self.width, self.height))
        dense_y = cv2.resize(grid_filled_y, (self.width, self.height))
        return np.dstack([dense_x, dense_y])

# ==========================================
# 3. Pipeline (Analysis & Stabilization)
# ==========================================
class SteadyFlowPipeline:
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_path = output_path
        
        self.flow_estimator = CeLiuFlowEstimator(self.width, self.height)
        self.motion_completer = SparseMotionCompleter(self.width, self.height)
        
        self.iters = 5 
        self.lambda_t = 100.0 
        self.radius = 30

    def get_spatial_outliers(self, flow):
        dx = np.gradient(flow[:,:,0], axis=1)
        dy = np.gradient(flow[:,:,1], axis=0)
        grad_mag = np.sqrt(dx**2 + dy**2)
        return grad_mag > 0.1

    def get_temporal_outliers(self, accumulated_flow):
        smoothed = gaussian_filter1d(accumulated_flow, sigma=3.0, axis=0)
        diff = np.linalg.norm(accumulated_flow - smoothed, axis=3)
        return diff > 0.2

    def stabilize_pixel_profiles(self, C):
        # Jacobi Iteration (Eq 4)
        P = np.copy(C)
        sigma = self.radius / 3.0
        k_size = int(self.radius * 2 + 1)
        window = cv2.getGaussianKernel(k_size, sigma).reshape(-1)
        window[self.radius] = 0
        w_sum = np.sum(window)
        
        for _ in range(10):
            Px, Py = P[:,:,:,0], P[:,:,:,1]
            Sum_Px = gaussian_filter1d(Px, sigma=sigma, axis=0, mode='nearest', truncate=3.0) * w_sum
            Sum_Py = gaussian_filter1d(Py, sigma=sigma, axis=0, mode='nearest', truncate=3.0) * w_sum
            
            gamma = 1.0 + self.lambda_t * w_sum
            
            # FIXED: Use np.stack with axis=3 to get proper shape
            Sum_P = np.stack((Sum_Px, Sum_Py), axis=3)
            
            P = (C + self.lambda_t * Sum_P) / gamma
        return P

    def run(self):
        # --- Step 1: Initialization (Raw Flow) ---
        print("1. Estimating Raw Optical Flow (Ce Liu Method via pyflow)...")
        raw_flows = []
        ret, prev = self.cap.read()
        
        # Handle empty video case
        if not ret:
            print("Error: Could not read first frame.")
            return
            
        # Dummy first flow (Time 0 has no motion)
        raw_flows.append(np.zeros((self.height, self.width, 2), dtype=np.float32))
        
        # Iterate, but respect the actual video length
        for _ in tqdm.tqdm(range(self.num_frames - 1)):
            ret, curr = self.cap.read()
            if not ret: 
                break # Video ended earlier than expected
            
            # Compute flow
            flow = self.flow_estimator.compute_flow(prev, curr)
            raw_flows.append(flow)
            prev = curr
            
        raw_flows = np.array(raw_flows)
        
        # --- CRITICAL FIX: Update num_frames to actual length ---
        self.num_frames = len(raw_flows)
        print(f"Actual frames processed: {self.num_frames}")
        
        # --- Step 2: Iterative Refinement ---
        C = np.cumsum(raw_flows, axis=0)
        SteadyFlow = np.copy(raw_flows)
        
        print("2. Iterative Refinement (Analysis -> Completion -> Stabilization)...")
        for iteration in range(self.iters):
            print(f"  Iteration {iteration + 1}/{self.iters}")
            
            mask_outliers = np.zeros((self.num_frames, self.height, self.width), bool)
            
            # A. Discontinuity Identification
            if iteration == 0:
                print("    Applying Spatial Analysis...")
                for t in range(self.num_frames):
                    mask_outliers[t] = self.get_spatial_outliers(SteadyFlow[t])
            else:
                print("    Applying Temporal Analysis...")
                C_current = np.cumsum(SteadyFlow, axis=0)
                mask_outliers = self.get_temporal_outliers(C_current)
                
            total_mask = mask_outliers 
            
            # B. Motion Completion (Fill Holes)
            for t in tqdm.tqdm(range(self.num_frames), desc="    Filling Holes"):
                if np.any(total_mask[t]):
                    SteadyFlow[t] = self.motion_completer.fill_holes(raw_flows[t], total_mask[t])
                else:
                    SteadyFlow[t] = raw_flows[t]
            
            # C. Stabilization (Pixel Profiles)
            C = np.cumsum(SteadyFlow, axis=0)
            P = self.stabilize_pixel_profiles(C) # Pass C, not SteadyFlow
            
            # Update SteadyFlow for next iteration (derivative of smoothed path)
            SteadyFlow[1:] = np.diff(P, axis=0)
            SteadyFlow[0] = P[0]

        # --- Step 3: Final Rendering ---
        print("3. Rendering Final Video...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        
        for t in tqdm.tqdm(range(self.num_frames)):
            ret, frame = self.cap.read()
            if not ret: break
            
            # Update field B_t = P_t - C_t
            # Use original raw accumulation for C
            orig_C_t = np.sum(raw_flows[:t+1], axis=0)
            target_P_t = P[t]
            
            update = target_P_t - orig_C_t
            
            # Warping
            map_x = (grid_x - update[:,:,0]).astype(np.float32)
            map_y = (grid_y - update[:,:,1]).astype(np.float32)
            
            stab = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            writer.write(stab)
            
        writer.release()
        self.cap.release()
        print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Usage: python final_steadyflow.py input.mp4 output.mp4")
    else: SteadyFlowPipeline(sys.argv[1], sys.argv[2]).run()
