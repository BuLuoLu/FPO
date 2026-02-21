import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import maximum_filter
import torch
import torch.nn.functional as F
import random


def set_seed(seed=42):
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value, default is 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")

def extract_points_from_heatmap(heatmap, threshold=0.5, nms_kernel=3, use_subpixel=False):
    """
    Extract point locations from a heatmap using peak detection and NMS.
    
    This function detects local maxima in a heatmap and optionally refines
    their positions using subpixel accuracy through weighted averaging.
    
    Args:
        heatmap (np.ndarray or torch.Tensor): Input heatmap of shape [H, W]
        threshold (float): Minimum confidence threshold for peak detection (0-1), default 0.5
        nms_kernel (int): Kernel size for Non-Maximum Suppression (must be odd), default 3
        use_subpixel (bool): Whether to use subpixel refinement for better accuracy, default False
    
    Returns:
        tuple: A tuple containing:
            - points (list): List of detected point coordinates [(x, y), ...]
            - confidences (list): List of confidence scores for each point [conf, ...]
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    
    H, W = heatmap.shape
    
    local_max = maximum_filter(heatmap, size=nms_kernel)
    
    mask_local_max = (heatmap == local_max)
    mask_threshold = (heatmap > threshold)
    mask = mask_local_max & mask_threshold
    
    y_coords, x_coords = np.where(mask)
    
    points = []
    confidences = []
    
    for i in range(len(y_coords)):
        y, x = int(y_coords[i]), int(x_coords[i])
        
        confidence = float(heatmap[y, x])
        
        if use_subpixel:
            if 1 <= x < W-1 and 1 <= y < H-1:
                neighborhood = heatmap[y-1:y+2, x-1:x+2]
                
                weights = np.exp(neighborhood - neighborhood.max())
                weights = weights / weights.sum()
                
                dx = (weights[1, 2] - weights[1, 0])
                dy = (weights[2, 1] - weights[0, 1])
                
                x_sub = x + dx * 0.5 
                y_sub = y + dy * 0.5
                
                points.append((float(x_sub), float(y_sub)))
            else:
                points.append((float(x), float(y)))
        else:
            points.append((float(x), float(y)))
        
        confidences.append(confidence)
    
    return points, confidences

def extract_keypoints_with_offset(heatmap, offset, threshold=0.3, nms_kernel=5, downsample_ratio=4):
    max_filtered = maximum_filter(heatmap, size=nms_kernel)
    peaks = (heatmap == max_filtered) & (heatmap > threshold)
    
    y_indices, x_indices = np.where(peaks)
    scores = heatmap[y_indices, x_indices]
    
    keypoints = []
    for i in range(len(x_indices)):
        x_int = x_indices[i]
        y_int = y_indices[i]
        score = scores[i]
        
        offset_x = offset[0, y_int, x_int] 
        offset_y = offset[1, y_int, x_int]
        
        x_feat = x_int + offset_x 
        y_feat = y_int + offset_y  
        
        keypoints.append((x_feat, y_feat, score))
    
    return keypoints

def get_peak_points(heatmap, threshold=0.3, neighborhood_size=21):
    """
    Extract peak points from heatmap using simple local maximum detection.
    
    This is an alternative implementation to extract_points_from_heatmap,
    using a simpler brute-force approach for finding local maxima.
    
    Args:
        heatmap (np.ndarray): Input heatmap of shape [H, W]
        threshold (float): Minimum confidence threshold for peak detection, default 0.3
        neighborhood_size (int): Size of the local neighborhood to search, default 21
    
    Returns:
        list: List of detected peak coordinates [(x, y), ...]
    """ 
    peaks = [] 
    candidate_indices = np.argwhere(heatmap > threshold)
    for y, x in candidate_indices:
        y_min = max(0, y - neighborhood_size // 2) 
        y_max = min(heatmap.shape[0], y + neighborhood_size // 2 + 1) 
        x_min = max(0, x - neighborhood_size // 2) 
        x_max = min(heatmap.shape[1], x + neighborhood_size // 2 + 1)

        local_patch = heatmap[y_min:y_max, x_min:x_max]
        
        if heatmap[y, x] == local_patch.max(): 
            peaks.append((x, y))
            # Remove duplicates
            peaks = list(set(peaks))
    return peaks

def generate_oriented_gaussian(center, direction_unit_vector, window_size, sigma_major=None, sigma_minor=None):
    """
    Generate an oriented Gaussian kernel for directional object detection.
    
    Creates an anisotropic Gaussian kernel aligned with a specified direction,
    with the kernel masked to only include the forward-facing half (directional).
    This is particularly useful for modeling oriented objects like follicles.
    
    Args:
        center (tuple): Center point coordinates (x, y) in image space
        direction_unit_vector (np.ndarray): Unit vector [dx, dy] indicating orientation
        window_size (int): Size of the Gaussian kernel window (should be odd)
        sigma_major (float, optional): Standard deviation along major axis (direction).
                                       Default: vec_length / 1.5
        sigma_minor (float, optional): Standard deviation along minor axis (perpendicular).
                                       Default: sigma_major / 4.0
    
    Returns:
        tuple: A tuple containing:
            - gaussian (np.ndarray): Normalized Gaussian kernel of shape [window_size, window_size]
            - window_coords (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
    
    Note:
        - The Gaussian is masked to only include values in the forward direction
        - sigma_major and sigma_minor control the elongation of the Gaussian
        - The kernel is normalized to have maximum value of 1.0
    """
    vec_length = 50.0

    if sigma_major is None:
        sigma_major = vec_length / 1.5  
    if sigma_minor is None:
        sigma_minor = sigma_major / 4.0
    
    norm_dx = direction_unit_vector[0]
    norm_dy = direction_unit_vector[1]
    
    angle = np.arctan2(norm_dy, norm_dx)
    
    half_window = window_size // 2
    x = np.arange(-half_window, half_window + 1)
    y = np.arange(-half_window, half_window + 1)
    xx, yy = np.meshgrid(x, y)
    
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    xx_rot = xx * cos_angle - yy * sin_angle
    yy_rot = xx * sin_angle + yy * cos_angle
    
    gaussian = np.exp(-(xx_rot**2 / (2 * sigma_major**2) + yy_rot**2 / (2 * sigma_minor**2)))
    
    mask = xx_rot > 0
    gaussian = gaussian * mask 
    
    if gaussian.max() > 0:
        gaussian = gaussian / gaussian.max()
    
    cx, cy = int(center[0]), int(center[1])
    x_min = cx - half_window
    y_min = cy - half_window
    x_max = cx + half_window + 1
    y_max = cy + half_window + 1
    
    window_coords = (x_min, y_min, x_max, y_max)
    
    return gaussian, window_coords

def calculate_two_point_direction(point_start, point_end):
    """
    Calculate direction vector from two annotation points (for XML format annotations).
    
    This function computes the orientation of an object based on two points:
    typically the tip (p1) and the center of the base (midpoint of p2 and p3)
    of a triangle annotation.
    
    Args:
        point_start (array-like): Starting point [x, y], corresponds to triangle tip (p1)
        point_end (array-like): Ending point [x, y], corresponds to base center 
                               (midpoint between p2 and p3)
    
    Returns:
        tuple: A tuple containing:
            - p1 (np.ndarray): Center point (starting point) as float32 array
            - theta (float): Angle in radians (computed using atan2)
            - direction_unit_vector (np.ndarray): Unit direction vector [cos_theta, sin_theta]
    
    Note:
        - Direction is calculated from point_start towards point_end
        - Returns zero direction ([1, 0]) if points are identical
    """
    p1 = np.array(point_start, dtype=np.float32)
    p_end = np.array(point_end, dtype=np.float32)
    
    direction_vector = p_end - p1
    vec_length = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
    
    if vec_length > 0:
        theta = np.arctan2(direction_vector[1], direction_vector[0])
        cos_theta = direction_vector[0] / vec_length
        sin_theta = direction_vector[1] / vec_length
    else:
        theta = 0.0
        cos_theta = 1.0
        sin_theta = 0.0
    
    direction_unit_vector = np.array([cos_theta, sin_theta])
    return p1, theta, direction_unit_vector