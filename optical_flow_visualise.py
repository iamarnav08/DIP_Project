import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.models.optical_flow import raft_large
from matplotlib.colors import hsv_to_rgb
import os # Keep os import for file path checking


# --- 1. Utility Function to Extract a Specific Frame ---

def get_frame_by_number(video_path, frame_num):
    """
    Reads a specific frame from a video file.
    Note: Frame indexing starts at 1 for typical video processing, 
          but OpenCV's CAP_PROP_POS_FRAMES is 0-indexed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file at {video_path}")

    # Set the frame position (0-indexed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    
    success, frame = cap.read()
    cap.release()
    
    if not success:
        raise ValueError(f"Error: Failed to read frame number {frame_num} from video.")
        
    return frame # Frame is in BGR format


# --- 2. Flow Visualization and Computation Functions (Unchanged from previous response) ---

def flow_to_colorwheel_image(flow):
    """
    Convert a 2-channel optical flow array (dx, dy) to a color-coded image 
    using the standard HSV color wheel (Baker et al. / MPI-Sintel).
    """
    h, w, _ = flow.shape
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Hue: angle (0 to 2*pi) maps to 0-180 for OpenCV
    hsv[..., 0] = ang * 90 / np.pi 
    hsv[..., 1] = 255
    
    max_mag = mag.max()
    if max_mag > 0:
        norm_mag = mag / max_mag
    else:
        norm_mag = mag

    hsv[..., 2] = np.clip(norm_mag * 255, 0, 255).astype(np.uint8)
    
    bgr_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    return rgb_image


def compute_optical_flow(frame1, frame2, model, device):
    """Compute optical flow between two frames using RAFT model."""
    # Convert from BGR (OpenCV default) to RGB for torchvision model
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    frame1_tensor = F.to_tensor(frame1_rgb).unsqueeze(0).to(device)
    frame2_tensor = F.to_tensor(frame2_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        flow = model(frame1_tensor, frame2_tensor)[0]

    # Convert to NumPy array in (H, W, 2) format for visualization
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    return flow


def save_optical_flow_visualisation(frame1_rgb, flow_np, output_path):
    """Visualize and save the original frame and the computed flow."""
    flowImage = flow_to_colorwheel_image(flow_np)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(frame1_rgb)
    plt.title("Reference Frame")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(flowImage)
    plt.title("Optical Flow (Color-Coded)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Optical flow visualization saved to: {output_path}")

# --- 3. Main Execution Block (Modified) ---

if __name__ == "__main__":
    # --- Input Parameters ---
    VIDEO_PATH = '6.avi' # <<< CHANGE THIS to your video file path
    FRAME_NUM_1 = 57          # <<< First frame number (e.g., frame_0006)
    FRAME_NUM_2 = 58          # <<< Second frame number (e.g., frame_0007)
    OUTPUT_IMAGE_PATH = "optical_flow_result.png"  # Output image file

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'. Please update the VIDEO_PATH variable.")
        exit()

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the RAFT model once
    flow_model = raft_large(pretrained=True, progress=True).eval().to(device)

    try:
        # 1. Load the frames by number
        print(f"Loading frames {FRAME_NUM_1} and {FRAME_NUM_2}...")
        frame1_bgr = get_frame_by_number(VIDEO_PATH, FRAME_NUM_1)
        frame2_bgr = get_frame_by_number(VIDEO_PATH, FRAME_NUM_2)
        
        # 2. Compute Optical Flow
        print("Computing optical flow...")
        flow_np = compute_optical_flow(frame1_bgr, frame2_bgr, flow_model, device)
        
        # 3. Visualize and save
        print("Saving results...")
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        save_optical_flow_visualisation(frame1_rgb, flow_np, OUTPUT_IMAGE_PATH)
        
    except (IOError, ValueError) as e:
        print(f"An error occurred: {e}")

    print(f"Flow field computed. Shape: {flow_np.shape}")
