import cv2
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter1d

class SteadyFlowStabilizer:
    '''
    A SteadyFlowStabilizer stabilizes videos using the SteadyFlow algorithm:
    Spatially Smooth Optical Flow for Video Stabilization (Liu et al. CVPR 2014).
    
    It operates on every pixel (Dense Flow) rather than a sparse mesh.
    '''

    def __init__(self, visualize=False, num_iterations=5):
        self.visualize = visualize
        self.num_iterations = num_iterations  # Number of refinement iterations
        
        # SteadyFlow Parameters
        self.SMOOTHING_RADIUS = 30  # Temporal window size
        self.OUTLIER_THRESHOLD = 0.1 # Gradient threshold for discontinuities
        self.LAMBDA = 5.0 # Smoothing strength
        
    def stabilize(self, input_path, output_path):
        '''
        Main pipeline: Read -> Compute Flow -> Iterative Refinement -> Render
        '''
        # 1. Read Input Frames
        frames, fps, size = self._read_frames(input_path)
        h, w = size
        num_frames = len(frames)

        if num_frames < 2:
            print("Video too short to stabilize.")
            return

        # 2. Compute Dense Optical Flow (Original Motion C_t)
        raw_flows = self._compute_dense_optical_flow(frames)
        accumulated_raw = np.cumsum(raw_flows, axis=0)

        # 3. Iterative Refinement Loop
        print(f"\n=== Starting Iterative Refinement ({self.num_iterations} iterations) ===")
        
        current_accumulated = accumulated_raw.copy()
        
        for iteration in range(self.num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} ---")
            
            # 3a. Motion Completion (Handle Outliers)
            print(f"  Step 1: Motion Completion...")
            cleaned_accumulated = self._motion_completion(current_accumulated)

            # 3b. Smooth Trajectories (Stabilized Motion P_t)
            print(f"  Step 2: Smoothing Trajectories...")
            stabilized_accumulated = self._smooth_trajectories(cleaned_accumulated)
            
            # 3c. Update for next iteration
            # Use the stabilized result as input for next iteration
            current_accumulated = stabilized_accumulated.copy()
            
            print(f"  Iteration {iteration + 1} complete.")

        # 4. Calculate Final Update Field (B_t)
        # B_t = Stabilized - Original
        update_field = current_accumulated - accumulated_raw

        # 5. Render Output
        print("\n=== Rendering Final Video ===")
        self._render_video(frames, update_field, fps, output_path)
        
        if self.visualize:
            print("Preview generation skipped in batch mode.")

    def _read_frames(self, input_path):
        cap = cv2.VideoCapture(input_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm.tqdm(total=total, desc='Reading frames') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
                pbar.update(1)
        cap.release()
        return frames, fps, (h, w)

    def _compute_dense_optical_flow(self, frames):
        '''
        Compute dense optical flow for every pixel between consecutive frames.
        Returns an array of flows (N, H, W, 2).
        '''
        n = len(frames)
        h, w = frames[0].shape[:2]
        
        # We pad with one zero-frame at the start for accumulation consistency
        flows = np.zeros((n, h, w, 2), dtype=np.float32)
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Optimized Farneback Parameters for stability
        farneback_params = dict(
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        with tqdm.tqdm(total=n-1, desc='Computing Optical Flow') as pbar:
            for i in range(1, n):
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, **farneback_params
                )
                
                flows[i] = flow
                prev_gray = curr_gray
                pbar.update(1)
                
        return flows

    def _motion_completion(self, accumulated_flow):
        '''
        Identify spatial discontinuities and inpaint them.
        This prevents "tearing" at object boundaries.
        '''
        n, h, w, _ = accumulated_flow.shape
        cleaned = accumulated_flow.copy()
        
        with tqdm.tqdm(total=n, desc='    Motion Completion (Inpainting)', leave=False) as pbar:
            for i in range(n):
                flow = cleaned[i]
                
                # 1. Calculate Gradient Magnitude (Spatial Derivative)
                dx, dy = np.gradient(flow, axis=(0, 1))
                magnitude = np.sqrt(dx**2 + dy**2)
                mag_sum = np.sum(magnitude, axis=2)
                
                # 2. Create Mask
                mask = (mag_sum > self.OUTLIER_THRESHOLD).astype(np.uint8)
                
                # Dilate mask slightly to cover edges completely
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # 3. Inpaint
                if np.sum(mask) > 0:
                    blurred_flow = cv2.blur(flow, (15, 15))
                    mask_bool = mask > 0
                    cleaned[i, mask_bool] = blurred_flow[mask_bool]

                pbar.update(1)
                
        return cleaned

    def _smooth_trajectories(self, accumulated_flow):
        '''
        Smooth the accumulated flow over the temporal axis (Axis 0).
        This is equivalent to smoothing pixel profiles.
        '''
        # Gaussian Filter along axis 0 (Time)
        smoothed = gaussian_filter1d(accumulated_flow, sigma=self.LAMBDA, axis=0)
        
        # Anchor Constraint:
        # Blend back slightly towards the original to prevent drift
        alpha = 0.90 # Trust smooth path 90%
        stabilized = alpha * smoothed + (1 - alpha) * accumulated_flow
        
        return stabilized

    def _render_video(self, frames, update_field, fps, output_path):
        '''
        Warp frames based on the update field.
        update_field = Stabilized_Pos - Original_Pos
        '''
        h, w = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Create Base Grid (Float32 for remap)
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        with tqdm.tqdm(total=len(frames), desc='Rendering Video') as pbar:
            for i in range(len(frames)):
                frame = frames[i]
                shift = update_field[i] # (H, W, 2) -> (dx, dy)
                
                map_x = grid_x - shift[:,:,0]
                map_y = grid_y - shift[:,:,1]
                
                # Remap
                warped = cv2.remap(
                    frame, map_x, map_y, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                out.write(warped)
                pbar.update(1)
                
        out.release()
        print(f"\nVideo saved to {output_path}")

if __name__ == '__main__':
    import sys
    
    input_video = "input.mp4"
    output_video = "steadyflow_output.mp4"
    num_iterations = 5  # Default number of iterations
    
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    if len(sys.argv) > 2:
        output_video = sys.argv[2]
    if len(sys.argv) > 3:
        num_iterations = int(sys.argv[3])
        
    stabilizer = SteadyFlowStabilizer(visualize=False, num_iterations=num_iterations)
    stabilizer.stabilize(input_video, output_video)
