import cv2
import math
import numpy as np
import tqdm

class SteadyFlowStabilizer:
    '''
    A SteadyFlowStabilizer stabilizes videos using the SteadyFlow algorithm outlined in
    "SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization" by Shuaicheng Liu et al. (CVPR 2014).
    '''

    def __init__(self, 
                 temporal_smoothing_radius=7, 
                 spatial_smoothness_sigma=0.1,
                 motion_inpaint_iterations=10,
                 smoothing_iterations=28,
                 downscale_factor=0.2,
                 color_outside_image_area_bgr=(0, 0, 0)):
        '''
        Constructor.

        Input:
        * temporal_smoothing_radius: The radius of the temporal window for smoothing pixel profiles.
        * spatial_smoothness_sigma: Sigma for Gaussian used in discontinuity identification.
        * motion_inpaint_iterations: Number of diffusion iterations to fill in missing motion regions.
        * smoothing_iterations: Number of iterations for the iterative Jacobi-based smoothing.
        * downscale_factor: Factor to resize frames for flow calculation (0.5 = half size). 
                            SteadyFlow is dense and heavy; downscaling speeds it up significantly 
                            with minimal quality loss.
        * color_outside_image_area_bgr: Color for border regions.
        '''
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.spatial_smoothness_sigma = spatial_smoothness_sigma
        self.motion_inpaint_iterations = motion_inpaint_iterations
        self.smoothing_iterations = smoothing_iterations
        self.downscale_factor = downscale_factor
        self.color_outside_image_area_bgr = color_outside_image_area_bgr

    def stabilize(self, input_path, output_path):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path using SteadyFlow.
        '''
        
        # 1. Read Frames
        frames, fps = self._read_frames(input_path)
        if not frames:
            print("No frames found.")
            return

        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        print(f"Video Loaded: {num_frames} frames, {w}x{h}.")

        # 2. Estimate SteadyFlow (Dense Optical Flow + Outlier Removal)
        print("Estimating SteadyFlow (Dense Optical Flow & Motion Completion)...")
        # steady_flows[t] contains flow from frame t to t+1
        # shape: (num_frames-1, h, w, 2)
        steady_flows = self._estimate_steady_flows(frames)

        # 3. Build Pixel Profiles (Accumulated Motion)
        # C_t(p) = sum_{k=0}^{t-1} u_k(p)
        # We perform cumsum along time axis.
        # Shape: (num_frames, h, w, 2)
        print("Building Pixel Profiles...")
        
        # Initialize accumulated motion fields
        # flow_acc[t] is the absolute displacement of pixel p from frame 0 to frame t
        # flow_acc[0] is all zeros
        # flow_acc[1] is steady_flows[0]
        # flow_acc[2] is steady_flows[0] + steady_flows[1]
        flow_acc = np.zeros((num_frames, h, w, 2), dtype=np.float32)
        flow_acc[1:] = np.cumsum(steady_flows, axis=0)

        # 4. Smooth Pixel Profiles
        print("Smoothing Pixel Profiles...")
        smooth_flow_acc = self._smooth_pixel_profiles(flow_acc)

        # 5. Render Stabilized Video
        # The warp field B_t = P_t - C_t
        # Warping: stabilized_frame(x) = original_frame(x + B_t(x))
        # Note: In opencv remap, map_x(x,y) = x + dx, map_y(x,y) = y + dy
        print("Rendering Stabilized Video...")
        warp_fields = smooth_flow_acc - flow_acc
        
        self._write_stabilized_video(output_path, frames, warp_fields, fps)
        
        return 0, 0, 0 # Placeholder for scores

    def _read_frames(self, input_path):
        video = cv2.VideoCapture(input_path)
        frames = []
        fps = video.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        return frames, fps

    def _estimate_steady_flows(self, frames):
        num_frames = len(frames)
        h, w = frames[0].shape[:2]
        
        # Resize dimensions for flow calculation
        process_w = int(w * self.downscale_factor)
        process_h = int(h * self.downscale_factor)
        
        # Storage for flows (we will resize them back to original size later if needed, 
        # or keep them dense but low-res and interpolate during rendering. 
        # For quality, we upscale flow to original resolution immediately).
        flows = np.zeros((num_frames - 1, h, w, 2), dtype=np.float32)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (process_w, process_h))

        for t in tqdm.tqdm(range(num_frames - 1)):
            curr_gray = cv2.cvtColor(frames[t+1], cv2.COLOR_BGR2GRAY)
            curr_gray_resized = cv2.resize(curr_gray, (process_w, process_h))

            # A. Initialization: Global Homography + Residual Optical Flow
            # (Simplified: We rely on Farneback to capture both global and local, 
            # as it handles large displacements reasonably well with pyramid).
            # The paper suggests Init = Homography + Residual. 
            # For this implementation, we use robust dense flow directly.
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray_resized, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, 
                                                flags=0)
            
            # Upscale flow to original resolution
            # Note: Flow values must be scaled by (1/downscale_factor)
            flow_full = cv2.resize(flow, (w, h)) * (1.0 / self.downscale_factor)

            # B. Discontinuity Identification (Outlier Detection)
            # Spatial analysis: Gradient magnitude
            # Calculate gradient of flow magnitude
            dx = cv2.Sobel(flow_full, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow_full, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(dx**2 + dy**2)
            grad_mag_sum = grad_mag[:,:,0] + grad_mag[:,:,1] # Combine x and y components
            
            # Paper threshold: 0.1 (pixels?). Dependent on resolution. 
            # We use a heuristic relative threshold.
            spatial_mask = grad_mag_sum > 2.0 # Threshold for discontinuity
            
            # (Optional) Temporal analysis requires the accumulated flow, which we don't have fully yet.
            # We proceed with spatial outlier removal.
            
            # C. Motion Completion
            # We treat 'spatial_mask' as missing regions.
            # We fill them using diffusion (approximated by repeated blurring of masked regions).
            # This is faster than solving the sparse linear system described in the paper
            # but achieves the goal of enforcing spatial smoothness.
            
            clean_flow = flow_full.copy()
            
            # Simple diffusion strategy for inpainting float arrays
            for _ in range(self.motion_inpaint_iterations):
                # Blur the whole flow field
                blurred = cv2.blur(clean_flow, (15, 15))
                # Only update the outlier regions with the blurred (neighbors average) value
                clean_flow[spatial_mask] = blurred[spatial_mask]

            flows[t] = clean_flow
            prev_gray = curr_gray_resized

        return flows

    def _smooth_pixel_profiles(self, C):
        '''
        Iterative smoothing of pixel profiles.
        Eq (4): P_t^(xi+1) = 1/gamma * (C_t + lambda * sum(w * P_r^(xi)))
        
        This is essentially a weighted average over the temporal window.
        To implement efficiently without looping pixels, we use 1D convolution 
        along the time axis.
        '''
        num_frames, h, w, _ = C.shape
        P = C.copy() # Initialization P^0 = C
        
        # Lambda parameter from paper (controls smoothing strength)
        lam = 20.0 
        
        # Construct weights for the temporal window
        # Paper uses adaptive window. Here we use a fixed Gaussian window for robustness.
        # w_r = exp(-distance^2 / sigma^2)
        radius = self.temporal_smoothing_radius
        window_size = 2 * radius + 1
        sigma = radius / 3.0
        
        weights = np.exp(-np.square(np.arange(-radius, radius + 1)) / (2 * sigma * sigma))
        weights[radius] = 0 # w_{t,t} = 0 in the summation term of Eq 4
        
        # Calculate gamma (normalization factor)
        # gamma = 1 + lambda * sum(weights)
        gamma = 1 + lam * np.sum(weights)
        
        # We can implement the summation sum(w * P) as a 1D convolution over the time axis.
        # Since we need to do this for every pixel (H, W), we reshape to (T, H*W*2) 
        # to speed up or use scipy ndimage.
        
        # However, for memory efficiency with large videos, we process chunks or just iterate.
        # Given the "Iterative Refinement", we run the update loop.
        
        # Pre-compute convolution kernel
        # Kernel shape needs to be (window_size, 1, 1, 1) to broadcast over H, W, Ch
        kernel = weights.reshape(-1, 1, 1, 1)
        
        print("Running Iterative Smoothing...")
        for it in range(self.smoothing_iterations):
            # Compute weighted sum of neighbors
            # Pad P for convolution
            P_padded = np.pad(P, ((radius, radius), (0,0), (0,0), (0,0)), mode='edge')
            
            # Convolve along axis 0 (Time)
            # We manually implement the sliding window sum for memory efficiency or use a loop
            weighted_sum = np.zeros_like(P)
            
            # Vectorized sliding window
            # This loop is over the window radius (e.g., 20 iterations), not frames.
            for r in range(window_size):
                offset = r - radius
                if offset == 0: continue
                
                w_val = weights[r]
                # Shift P by offset
                # If offset is -1 (prev frame), we want P[t-1]
                # Slicing:
                if offset < 0:
                    # shift P "right" in time: P[0...T-1-off]
                    shifted = P_padded[radius + offset : num_frames + radius + offset]
                else:
                    shifted = P_padded[radius + offset : num_frames + radius + offset]
                
                weighted_sum += w_val * shifted

            # Update P
            P = (C + lam * weighted_sum) / gamma
            
        return P

    def _write_stabilized_video(self, output_path, frames, warp_fields, fps):
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Create meshgrid for remap
        # map_x[y,x] = x, map_y[y,x] = y
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for t in tqdm.tqdm(range(len(frames))):
            frame = frames[t]
            
            # Warp field B_t = P_t - C_t
            # This vector tells us how much to move the pixel to stabilize it.
            # If a pixel at (x,y) moved by (u,v) in the stabilization field,
            # we want to sample from (x+u, y+v).
            
            flow = warp_fields[t] # shape (h, w, 2)
            
            # flow is (dx, dy)
            map_x = grid_x - flow[:,:,0]
            map_y = grid_y - flow[:,:,1]
            
            stabilized_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderValue=self.color_outside_image_area_bgr)
            out.write(stabilized_frame)
            
        out.release()
        print(f"Stabilized video saved to {output_path}")

def main():
    # Example Usage
    # input_path = '/home/arnavsharma/Arnav/UG_3.1/DIP/Project/meshflow/video_2.mp4'
    input_path = './videos/video-1/video-1.m4v'
    output_path = './output_steadyflow5.mp4'
    
    # Initialize SteadyFlow Stabilizer
    # downscale_factor=0.5 processes flow at half resolution for speed
    stabilizer = SteadyFlowStabilizer(downscale_factor=0.5)
    
    stabilizer.stabilize(input_path, output_path)

if __name__ == '__main__':
    main()
