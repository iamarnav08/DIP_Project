# This one uses the equation solving method as given in the paper.

import cv2
import math
import numpy as np
import tqdm

class SteadyFlowStabilizer:
    '''
    A SteadyFlowStabilizer stabilizes videos using the SteadyFlow algorithm outlined in
    "SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization" by Shuaicheng Liu et al. (CVPR 2014).
    Enhanced with Jacobi energy optimization from MeshFlow.
    '''

    def __init__(self, 
                 temporal_smoothing_radius=7, 
                 spatial_smoothness_sigma=0.1,
                 motion_inpaint_iterations=10,
                 smoothing_iterations=28,
                 downscale_factor=0.2,
                 optimization_num_iterations=100,
                 adaptive_weight_scale=20.0,
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
        * optimization_num_iterations: Number of Jacobi iterations for energy minimization.
        * adaptive_weight_scale: Scale factor for adaptive weights (lambda in energy function).
        * color_outside_image_area_bgr: Color for border regions.
        '''
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.spatial_smoothness_sigma = spatial_smoothness_sigma
        self.motion_inpaint_iterations = motion_inpaint_iterations
        self.smoothing_iterations = smoothing_iterations
        self.downscale_factor = downscale_factor
        self.optimization_num_iterations = optimization_num_iterations
        self.adaptive_weight_scale = adaptive_weight_scale
        self.color_outside_image_area_bgr = color_outside_image_area_bgr

    def stabilize(self, input_path, output_path):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path using SteadyFlow with Jacobi energy optimization.
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
        steady_flows, homographies = self._estimate_steady_flows(frames)

        # 3. Build Pixel Profiles (Accumulated Motion)
        print("Building Pixel Profiles...")
        flow_acc = np.zeros((num_frames, h, w, 2), dtype=np.float32)
        flow_acc[1:] = np.cumsum(steady_flows, axis=0)

        # 4. Smooth Pixel Profiles using Jacobi Energy Optimization
        print("Smoothing Pixel Profiles with Jacobi Energy Optimization...")
        smooth_flow_acc = self._smooth_pixel_profiles_jacobi(flow_acc, homographies, w, h)

        # 5. Render Stabilized Video
        print("Rendering Stabilized Video...")
        warp_fields = smooth_flow_acc - flow_acc
        
        self._write_stabilized_video(output_path, frames, warp_fields, fps)
        
        return 0, 0, 0

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
        
        process_w = int(w * self.downscale_factor)
        process_h = int(h * self.downscale_factor)
        
        flows = np.zeros((num_frames - 1, h, w, 2), dtype=np.float32)
        homographies = np.zeros((num_frames, 3, 3), dtype=np.float32)
        homographies[-1] = np.identity(3)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (process_w, process_h))

        for t in tqdm.tqdm(range(num_frames - 1)):
            curr_gray = cv2.cvtColor(frames[t+1], cv2.COLOR_BGR2GRAY)
            curr_gray_resized = cv2.resize(curr_gray, (process_w, process_h))

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray_resized, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, 
                                                flags=0)
            
            flow_full = cv2.resize(flow, (w, h)) * (1.0 / self.downscale_factor)

            # Compute homography for adaptive weights
            early_features, late_features, homography = self._get_matched_features_and_homography(
                frames[t], frames[t+1]
            )
            if homography is not None:
                homographies[t] = homography
            else:
                homographies[t] = np.identity(3)

            # Discontinuity identification
            dx = cv2.Sobel(flow_full, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(flow_full, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(dx**2 + dy**2)
            grad_mag_sum = grad_mag[:,:,0] + grad_mag[:,:,1]
            
            spatial_mask = grad_mag_sum > 2.0
            
            # Motion completion
            clean_flow = flow_full.copy()
            for _ in range(self.motion_inpaint_iterations):
                blurred = cv2.blur(clean_flow, (15, 15))
                clean_flow[spatial_mask] = blurred[spatial_mask]

            flows[t] = clean_flow
            prev_gray = curr_gray_resized

        return flows, homographies

    def _get_matched_features_and_homography(self, early_frame, late_frame):
        '''
        Detect and match features between two frames to compute homography.
        '''
        feature_detector = cv2.FastFeatureDetector_create()
        
        early_gray = cv2.cvtColor(early_frame, cv2.COLOR_BGR2GRAY)
        late_gray = cv2.cvtColor(late_frame, cv2.COLOR_BGR2GRAY)
        
        early_keypoints = feature_detector.detect(early_gray)
        if len(early_keypoints) < 4:
            return None, None, None
        
        early_features = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])
        late_features, matched, _ = cv2.calcOpticalFlowPyrLK(early_gray, late_gray, early_features, None)
        
        if matched is None:
            return None, None, None
            
        matched_mask = matched.flatten().astype(dtype=bool)
        early_features = early_features[matched_mask]
        late_features = late_features[matched_mask]
        
        if len(early_features) < 4:
            return None, None, None
        
        homography, _ = cv2.findHomography(early_features, late_features, method=cv2.RANSAC)
        
        return early_features, late_features, homography

    def _smooth_pixel_profiles_jacobi(self, C, homographies, frame_width, frame_height):
        '''
        Smooth pixel profiles using Jacobi method with energy optimization.
        This combines temporal smoothing with adaptive weights based on frame motion.
        
        Minimizes energy function:
        E(P) = sum_t ||P_t - C_t||^2 + lambda_t * sum_r w_{t,r} ||P_t - P_r||^2
        '''
        num_frames, h, w, _ = C.shape
        
        # Compute adaptive weights from homographies
        adaptive_weights = self._get_adaptive_weights(num_frames, frame_width, frame_height, homographies)
        
        # Initialize P
        P = C.copy()
        
        radius = self.temporal_smoothing_radius
        sigma = radius / 3.0
        
        # Precompute temporal weights (Gaussian)
        offsets = np.arange(-radius, radius + 1)
        temporal_weights = np.exp(-np.square(offsets) / (2 * sigma * sigma))
        temporal_weights[radius] = 0  # Exclude self
        
        print("Running Jacobi Energy Optimization...")
        for iteration in tqdm.tqdm(range(self.optimization_num_iterations)):
            P_new = np.zeros_like(P)
            
            for t in range(num_frames):
                # Compute weighted sum of neighbors
                weighted_sum = np.zeros((h, w, 2), dtype=np.float32)
                weight_sum = 0.0
                
                for offset_idx, offset in enumerate(offsets):
                    neighbor_t = t + offset
                    
                    # Skip out of bounds and self
                    if neighbor_t < 0 or neighbor_t >= num_frames or offset == 0:
                        continue
                    
                    w_tr = temporal_weights[offset_idx]
                    weighted_sum += w_tr * P[neighbor_t]
                    weight_sum += w_tr
                
                # Adaptive weight for this frame
                lam_t = adaptive_weights[t]
                
                # Jacobi update: P_new = (C + 2*lam*weighted_sum) / (1 + 2*lam*weight_sum)
                if weight_sum > 0:
                    P_new[t] = (C[t] + 2.0 * lam_t * weighted_sum) / (1.0 + 2.0 * lam_t * weight_sum)
                else:
                    P_new[t] = C[t]
            
            P = P_new
        
        return P

    def _get_adaptive_weights(self, num_frames, frame_width, frame_height, homographies):
        '''
        Calculate adaptive weights based on homography properties.
        Similar to MeshFlow's adaptive weight calculation.
        '''
        adaptive_weights = np.zeros(num_frames)
        
        for t in range(num_frames):
            H = homographies[t].copy()
            H_affine = H.copy()
            H_affine[2, :] = [0, 0, 1]
            
            # Translation component
            tx = H[0, 2] / frame_width
            ty = H[1, 2] / frame_height
            translation_mag = math.sqrt(tx**2 + ty**2)
            
            # Affine component (ratio of eigenvalues)
            eigenvalues = np.linalg.eigvals(H_affine)
            eigenvalue_mags = np.sort(np.abs(eigenvalues))
            if eigenvalue_mags[-1] > 1e-6:
                affine_ratio = eigenvalue_mags[-2] / eigenvalue_mags[-1]
            else:
                affine_ratio = 1.0
            
            # Linear model from MeshFlow paper
            weight_1 = -1.93 * translation_mag + 0.95
            weight_2 = 5.83 * affine_ratio - 4.88
            
            adaptive_weights[t] = max(min(weight_1, weight_2), 0) * self.adaptive_weight_scale
        
        return adaptive_weights

    def _write_stabilized_video(self, output_path, frames, warp_fields, fps):
        h, w = frames[0].shape[:2]
        
        # Add cropping to reduce border artifacts
        crop_ratio = 0.9
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        crop_x = (w - crop_w) // 2
        crop_y = (h - crop_h) // 2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
        
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        for t in tqdm.tqdm(range(len(frames))):
            frame = frames[t]
            flow = warp_fields[t]
            
            # Constrain maximum displacement to prevent extreme warping
            max_displacement = 50.0
            displacement_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            scale = np.minimum(1.0, max_displacement / (displacement_mag + 1e-6))
            constrained_flow = flow * scale[:, :, np.newaxis]
            
            map_x = grid_x - constrained_flow[:,:,0]
            map_y = grid_y - constrained_flow[:,:,1]
            
            stabilized_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, 
                                         borderValue=self.color_outside_image_area_bgr)
            
            # Crop to valid region
            cropped = stabilized_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            out.write(cropped)
            
        out.release()
        print(f"Stabilized video saved to {output_path}")

def main():
    input_path = './videos/video-1/video-1.m4v'
    output_path = './output_steadyflow_jacobi.mp4'
    
    stabilizer = SteadyFlowStabilizer(
        downscale_factor=0.5,
        optimization_num_iterations=20,  # Reduced from 100 for faster processing
        adaptive_weight_scale=10.0,
        temporal_smoothing_radius=5      # Reduced from 7 for speed
    )
    
    stabilizer.stabilize(input_path, output_path)

if __name__ == '__main__':
    main()
