import cv2
import math
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
import concurrent.futures
from functools import partial

# RAFT model (you'll need to install RAFT or use torchvision's implementation)
try:
    from torchvision.models.optical_flow import raft_large
except ImportError:
    print("Warning: torchvision RAFT not available. Please install torchvision >= 0.12")
    raft_large = None

class SteadyFlowStabilizer:
    '''
    A SteadyFlowStabilizer stabilizes videos using the SteadyFlow algorithm with RAFT optical flow.
    Enhanced with Jacobi energy optimization from MeshFlow and parallelization.
    '''

    def __init__(self, 
                 temporal_smoothing_radius=7, 
                 spatial_smoothness_sigma=0.1,
                 motion_inpaint_iterations=10,
                 smoothing_iterations=28,
                 downscale_factor=0.2,
                 optimization_num_iterations=100,
                 adaptive_weight_scale=20.0,
                 color_outside_image_area_bgr=(0, 0, 0),
                 use_gpu=True,
                 num_workers=None):
        '''
        Constructor with additional GPU and parallelization options.
        '''
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.spatial_smoothness_sigma = spatial_smoothness_sigma
        self.motion_inpaint_iterations = motion_inpaint_iterations
        self.smoothing_iterations = smoothing_iterations
        self.downscale_factor = downscale_factor
        self.optimization_num_iterations = optimization_num_iterations
        self.adaptive_weight_scale = adaptive_weight_scale
        self.color_outside_image_area_bgr = color_outside_image_area_bgr
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers or max(1, cpu_count() - 2)
        
        # Initialize RAFT model
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if raft_large is not None:
            self.raft_model = raft_large(pretrained=True).to(self.device)
            self.raft_model.eval()
        else:
            self.raft_model = None
            print("Warning: Using Farneback flow as fallback")
        
        # Preprocessing transform for RAFT
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def stabilize(self, input_path, output_path):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path using SteadyFlow with RAFT optical flow and parallelization.
        '''
        
        # 1. Read Frames
        frames, fps = self._read_frames(input_path)
        if not frames:
            print("No frames found.")
            return

        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        print(f"Video Loaded: {num_frames} frames, {w}x{h}.")

        # 2. Estimate SteadyFlow with RAFT (Parallelized)
        print("Estimating SteadyFlow with RAFT (Dense Optical Flow & Motion Completion)...")
        steady_flows, homographies = self._estimate_steady_flows_parallel(frames)

        # 3. Build Pixel Profiles (Accumulated Motion)
        print("Building Pixel Profiles...")
        flow_acc = np.zeros((num_frames, h, w, 2), dtype=np.float32)
        flow_acc[1:] = np.cumsum(steady_flows, axis=0)

        # 4. Smooth Pixel Profiles using Jacobi Energy Optimization (Parallelized)
        print("Smoothing Pixel Profiles with Parallelized Jacobi Energy Optimization...")
        smooth_flow_acc = self._smooth_pixel_profiles_jacobi_parallel(flow_acc, homographies, w, h)

        # 5. Render Stabilized Video (Parallelized)
        print("Rendering Stabilized Video...")
        warp_fields = smooth_flow_acc - flow_acc
        
        self._write_stabilized_video_parallel(output_path, frames, warp_fields, fps)
        
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

    def _compute_raft_flow(self, frame1, frame2):
        '''
        Compute optical flow using RAFT model.
        '''
        if self.raft_model is None:
            # Fallback to Farneback
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                              pyr_scale=0.5, levels=3, winsize=15, 
                                              iterations=3, poly_n=5, poly_sigma=1.2, 
                                              flags=0)
            return flow
        
        h, w = frame1.shape[:2]
        
        # Pad to make dimensions divisible by 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        frame1_padded = cv2.copyMakeBorder(frame1, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        frame2_padded = cv2.copyMakeBorder(frame2, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        
        # Convert frames to tensor
        with torch.no_grad():
            img1_tensor = self.preprocess(cv2.cvtColor(frame1_padded, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            img2_tensor = self.preprocess(cv2.cvtColor(frame2_padded, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            
            # Compute flow with RAFT
            flow_predictions = self.raft_model(img1_tensor, img2_tensor)
            flow_padded = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()
            
            # Remove padding from flow
            flow = flow_padded[:h, :w]
            
            return flow


    def _process_frame_pair(self, frame_data):
        '''
        Process a single frame pair for optical flow computation.
        '''
        t, frames = frame_data
        frame1 = frames[t]
        frame2 = frames[t + 1]
        
        h, w = frame1.shape[:2]
        process_w = int(w * self.downscale_factor)
        process_h = int(h * self.downscale_factor)
        
        # Resize frames for processing
        frame1_resized = cv2.resize(frame1, (process_w, process_h))
        frame2_resized = cv2.resize(frame2, (process_w, process_h))
        
        # Compute optical flow
        flow = self._compute_raft_flow(frame1_resized, frame2_resized)
        flow_full = cv2.resize(flow, (w, h)) * (1.0 / self.downscale_factor)

        # Compute homography for adaptive weights
        early_features, late_features, homography = self._get_matched_features_and_homography(
            frame1, frame2
        )
        if homography is None:
            homography = np.identity(3)

        # Discontinuity identification and motion completion
        clean_flow = self._motion_completion(flow_full)

        return t, clean_flow, homography

    def _motion_completion(self, flow_full):
        '''
        Apply motion completion to handle discontinuities.
        '''
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
            
        return clean_flow

    def _estimate_steady_flows_parallel(self, frames):
        '''
        Parallelize optical flow computation across frame pairs.
        '''
        num_frames = len(frames)
        h, w = frames[0].shape[:2]
        
        flows = np.zeros((num_frames - 1, h, w, 2), dtype=np.float32)
        homographies = np.zeros((num_frames, 3, 3), dtype=np.float32)
        homographies[-1] = np.identity(3)

        # Prepare frame pairs for parallel processing
        frame_pairs = [(t, frames) for t in range(num_frames - 1)]
        
        # Process in batches to manage GPU memory
        batch_size = 4 if self.use_gpu else self.num_workers
        
        with tqdm.tqdm(total=num_frames - 1, desc="Computing optical flow") as pbar:
            for i in range(0, len(frame_pairs), batch_size):
                batch = frame_pairs[i:i + batch_size]
                
                if self.use_gpu and self.raft_model is not None:
                    # Sequential processing for GPU to avoid memory issues
                    for frame_data in batch:
                        t, clean_flow, homography = self._process_frame_pair(frame_data)
                        flows[t] = clean_flow
                        homographies[t] = homography
                        pbar.update(1)
                else:
                    # CPU parallel processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        results = list(executor.map(self._process_frame_pair, batch))
                        
                    for t, clean_flow, homography in results:
                        flows[t] = clean_flow
                        homographies[t] = homography
                        pbar.update(1)

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

    def _jacobi_update_frame(self, args):
        '''
        Perform Jacobi update for a single frame.
        '''
        t, P, C, offsets, temporal_weights, adaptive_weights, num_frames, radius = args
        
        h, w = P.shape[1:3]
        
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
            P_new_t = (C[t] + 2.0 * lam_t * weighted_sum) / (1.0 + 2.0 * lam_t * weight_sum)
        else:
            P_new_t = C[t]
            
        return t, P_new_t

    def _smooth_pixel_profiles_jacobi_parallel(self, C, homographies, frame_width, frame_height):
        '''
        Parallelize Jacobi energy optimization across frames using threading (CPU-safe with GPU).
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
        
        print("Running Parallelized Jacobi Energy Optimization...")
        for iteration in tqdm.tqdm(range(self.optimization_num_iterations)):
            P_new = np.zeros_like(P)
            
            # Prepare arguments for parallel processing
            frame_args = [(t, P, C, offsets, temporal_weights, adaptive_weights, num_frames, radius) 
                        for t in range(num_frames)]
            
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor for GPU safety
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._jacobi_update_frame, frame_args))
            
            # Collect results
            for t, P_new_t in results:
                P_new[t] = P_new_t
            
            P = P_new
        
        return P

    def _get_adaptive_weights(self, num_frames, frame_width, frame_height, homographies):
        '''
        Calculate adaptive weights based on homography properties.
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

    def _warp_frame(self, args):
        '''
        Warp a single frame using the computed warp field.
        '''
        t, frame, warp_field, grid_x, grid_y, max_displacement, color_outside = args
        
        flow = warp_field
        
        # Constrain maximum displacement to prevent extreme warping
        displacement_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        scale = np.minimum(1.0, max_displacement / (displacement_mag + 1e-6))
        constrained_flow = flow * scale[:, :, np.newaxis]
        
        map_x = grid_x - constrained_flow[:,:,0]
        map_y = grid_y - constrained_flow[:,:,1]
        
        stabilized_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, 
                                   borderValue=color_outside)
        
        return t, stabilized_frame

    def _write_stabilized_video_parallel(self, output_path, frames, warp_fields, fps):
            '''
            Parallelize frame warping for video output using threading.
            '''
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
            max_displacement = 50.0
            
            # Prepare arguments for parallel processing
            frame_args = [(t, frames[t], warp_fields[t], grid_x, grid_y, max_displacement, 
                        self.color_outside_image_area_bgr) for t in range(len(frames))]
            
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(tqdm.tqdm(executor.map(self._warp_frame, frame_args), 
                                    total=len(frames), desc="Warping frames"))
            
            # Sort results by frame index and write to video
            results.sort(key=lambda x: x[0])
            for t, stabilized_frame in results:
                # Crop to valid region
                cropped = stabilized_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                out.write(cropped)
                
            out.release()
            print(f"Stabilized video saved to {output_path}")


def main():
    input_path = '.avi'
    output_path = './limit_result_raft_parallel3.mp4'
    
    stabilizer = SteadyFlowStabilizer(
        downscale_factor=0.5,
        optimization_num_iterations=100,
        adaptive_weight_scale=10.0,
        temporal_smoothing_radius=3,
        spatial_smoothness_sigma=0.1,
        motion_inpaint_iterations=10,
        smoothing_iterations=10,
        use_gpu=True,
        num_workers=8  # Adjust based on your CPU cores
    )
    
    stabilizer.stabilize(input_path, output_path)

if __name__ == '__main__':
    main()
