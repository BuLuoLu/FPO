import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
if __package__:
    from .utils import generate_oriented_gaussian
else:
    from utils import generate_oriented_gaussian


class HairFollicleDataset(Dataset):
    """
    PyTorch Dataset for hair follicle detection with orientation.
    
    This dataset generates multiple ground truth maps:
    1. Heatmap: Gaussian peaks at object centers
    2. Direction field: Unit vectors indicating object orientation
    3. Direction mask: Valid regions for direction supervision
    4. Offset map: Sub-pixel location refinement (CenterNet-style)
    """
    
    def __init__(self, 
                 image_dir, 
                 label_dir, 
                 image_size=(1024, 1024),
                 sigma_major=None,
                 sigma_minor=None,
                 train=True,
                 use_augmentation=True,
                 heatmap_downsample_factor=1,
                 file_list=None):
        """
        Initialize the HairFollicleDataset.
        
        Args:
            image_dir (str or Path): Directory containing input images (.jpg)
            label_dir (str or Path): Directory containing JSON annotation files
            image_size (tuple): Target image size (height, width), default (1024, 1024)
            sigma_major (float, optional): Standard deviation along major axis for Gaussian kernel
            sigma_minor (float, optional): Standard deviation along minor axis for Gaussian kernel
            train (bool): Whether this is training mode, default True
            use_augmentation (bool): Whether to apply data augmentation, default True
            heatmap_downsample_factor (int): Downsampling factor for output heatmap, default 1
            file_list (list, optional): List of filename stems (without extension) to include in dataset
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.sigma_major = sigma_major
        self.sigma_minor = sigma_minor
        self.train = train
        self.use_augmentation = use_augmentation
        self.heatmap_downsample_factor = heatmap_downsample_factor
        
        self.heatmap_size = (
            image_size[0] // heatmap_downsample_factor,
            image_size[1] // heatmap_downsample_factor
        )
        
        self.valid_samples = []
        
        if file_list is not None:
            for filename in file_list:
                img_file = self.image_dir / f"{filename}.jpg"
                label_file = self.label_dir / f"{filename}.json"
                
                if img_file.exists() and label_file.exists():
                    self.valid_samples.append({
                        'image': img_file,
                        'label': label_file
                    })
        else:
            image_files = sorted([f for f in self.image_dir.glob('*.jpg')])
            for img_file in image_files:
                label_file = self.label_dir / f"{img_file.stem}.json"
                
                if label_file.exists():
                    self.valid_samples.append({
                        'image': img_file,
                        'label': label_file
                    })
        
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """
        Create data augmentation and preprocessing transforms.
        
        Returns:
            albumentations.Compose: Composition of transforms
            
        Note:
            - Training mode includes: HorizontalFlip, VerticalFlip, BrightnessContrast
            - Both modes include: Resize, Normalize, ToTensor
            - Heatmap is registered as 'mask' type for proper augmentation
        """
        if self.train and self.use_augmentation:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'heatmap': 'mask'})
        else:
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'heatmap': 'mask'})
        
        return transform
    
    def _generate_heatmap_from_json(self, json_path, orig_height, orig_width):
        """
        Generate ground truth maps from JSON annotation file.
        
        Args:
            json_path (Path): Path to JSON annotation file
            orig_height (int): Original image height in pixels
            orig_width (int): Original image width in pixels
            
        Returns:
            tuple: A tuple containing:
                - heatmap (np.ndarray): Shape [H, W], Gaussian peaks at object centers
                - direction_field (np.ndarray): Shape [H, W, 2], direction vectors (cos, sin)
                - direction_mask (np.ndarray): Shape [H, W], binary mask for valid directions
                - offset_map (np.ndarray): Shape [H, W, 2], sub-pixel offset corrections
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        heatmap = np.zeros((orig_height, orig_width), dtype=np.float32)
        direction_field = np.zeros((orig_height, orig_width, 2), dtype=np.float32)
        direction_mask = np.zeros((orig_height, orig_width), dtype=np.float32)
        offset_map = np.zeros((orig_height, orig_width, 2), dtype=np.float32)
        
        detections = data.get('detections', [])
        
        for detection in detections:
            center = detection['center']
            cx, cy = float(center[0]), float(center[1])
            
            direction = detection['direction']
            cos_theta = float(direction['cos'])
            sin_theta = float(direction['sin'])
            
            direction_unit_vector = np.array([cos_theta, sin_theta], dtype=np.float32)
            p1 = np.array([cx, cy], dtype=np.float32)
            
            heatmap = self._add_oriented_gaussian_to_heatmap(
                heatmap, p1, direction_unit_vector, orig_width, orig_height
            )
            
            direction_field, direction_mask = self._add_direction_field(
                direction_field, direction_mask, cx, cy, cos_theta, sin_theta,
                orig_width, orig_height
            )
            
            offset_map = self._add_offset_to_map(
                offset_map, cx, cy, orig_width, orig_height
            )
        
        return heatmap, direction_field, direction_mask, offset_map
    
    def _add_oriented_gaussian_to_heatmap(self, heatmap, center, direction_vector, 
                                          orig_width, orig_height):
        """
        Add an oriented Gaussian kernel to the heatmap at the specified location.
        
        Args:
            heatmap (np.ndarray): Existing heatmap to add Gaussian to
            center (np.ndarray): Center point [x, y]
            direction_vector (np.ndarray): Direction unit vector [cos, sin]
            orig_width (int): Image width
            orig_height (int): Image height
            
        Returns:
            np.ndarray: Updated heatmap with Gaussian added
        """
        fixed_length = 50.0
        sigma_major = self.sigma_major if self.sigma_major is not None else fixed_length / 1.5
        half_window_radius = int(3 * sigma_major) + 10
        window_size = 2 * half_window_radius + 1
        
        gaussian, window_coords = generate_oriented_gaussian(
            center, direction_vector, window_size,
            sigma_major=self.sigma_major,
            sigma_minor=self.sigma_minor
        )
        
        x_min, y_min, x_max, y_max = window_coords
        
        valid_x_min = max(0, x_min)
        valid_y_min = max(0, y_min)
        valid_x_max = min(orig_width, x_max)
        valid_y_max = min(orig_height, y_max)
        
        gauss_x_start = valid_x_min - x_min
        gauss_y_start = valid_y_min - y_min
        gauss_x_end = gauss_x_start + (valid_x_max - valid_x_min)
        gauss_y_end = gauss_y_start + (valid_y_max - valid_y_min)
        
        heatmap[valid_y_min:valid_y_max, valid_x_min:valid_x_max] = np.maximum(
            heatmap[valid_y_min:valid_y_max, valid_x_min:valid_x_max],
            gaussian[gauss_y_start:gauss_y_end, gauss_x_start:gauss_x_end]
        )
        
        return heatmap
    
    def _add_direction_field(self, direction_field, direction_mask, cx, cy,
                            cos_theta, sin_theta, orig_width, orig_height):
        """
        Add direction vectors in a square region around the object center.
        
        Args:
            direction_field (np.ndarray): Existing direction field [H, W, 2]
            direction_mask (np.ndarray): Existing direction mask [H, W]
            cx (float): Center x-coordinate
            cy (float): Center y-coordinate
            cos_theta (float): Cosine of direction angle
            sin_theta (float): Sine of direction angle
            orig_width (int): Image width
            orig_height (int): Image height
            
        Returns:
            tuple: Updated (direction_field, direction_mask)
        """
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        radius = 5 
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny = cy_int + dy
                nx = cx_int + dx
                if 0 <= nx < orig_width and 0 <= ny < orig_height:
                    direction_field[ny, nx, 0] = cos_theta
                    direction_field[ny, nx, 1] = sin_theta
                    direction_mask[ny, nx] = 1.0
        
        return direction_field, direction_mask
    
    def _add_offset_to_map(self, offset_map, cx, cy, orig_width, orig_height):
        """
        Add sub-pixel offset at object center location (CenterNet-style).
        
        The offset represents the quantization error when mapping continuous
        coordinates to discrete feature map grids.
        
        Args:
            offset_map (np.ndarray): Existing offset map [H, W, 2]
            cx (float): Center x-coordinate in original image
            cy (float): Center y-coordinate in original image
            orig_width (int): Image width
            orig_height (int): Image height
            
        Returns:
            np.ndarray: Updated offset map
            
        Example:
            If cx = 100.3 and downsample_factor = 4:
            - cx_feat = 100.3 / 4 = 25.075
            - cx_int_feat = 25
            - offset_x = 0.075 (fractional part to be recovered)
        """
        cx_feat = cx / self.heatmap_downsample_factor
        cy_feat = cy / self.heatmap_downsample_factor
        
        cx_int_feat = int(cx_feat)
        cy_int_feat = int(cy_feat)
        
        offset_x = cx_feat - cx_int_feat
        offset_y = cy_feat - cy_int_feat
        
        cx_int_orig = int(round(cx))
        cy_int_orig = int(round(cy))
        
        if 0 <= cx_int_orig < orig_width and 0 <= cy_int_orig < orig_height:
            offset_map[cy_int_orig, cx_int_orig, 0] = offset_x
            offset_map[cy_int_orig, cx_int_orig, 1] = offset_y
        
        return offset_map
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        image = cv2.imread(str(sample['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        
        heatmap, direction_field, direction_mask, offset_map = self._generate_heatmap_from_json(
            sample['label'], 
            orig_height, 
            orig_width
        )
        
        augmented = self.transform(image=image, heatmap=heatmap)
        
        image_tensor = augmented['image'] 
        heatmap_tensor = augmented['heatmap']  
        
        if isinstance(heatmap_tensor, np.ndarray):
            heatmap_tensor = torch.from_numpy(heatmap_tensor).float()
        else:
            heatmap_tensor = heatmap_tensor.float()
        
        if self.heatmap_downsample_factor > 1:
            heatmap_np = heatmap_tensor.numpy() if isinstance(heatmap_tensor, torch.Tensor) else heatmap_tensor
            heatmap_downsampled = cv2.resize(
                heatmap_np,
                (self.heatmap_size[1], self.heatmap_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            heatmap_tensor = torch.from_numpy(heatmap_downsampled).float()
        
        direction_field_resized = cv2.resize(
            direction_field, 
            (self.heatmap_size[1], self.heatmap_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        magnitude = np.sqrt(direction_field_resized[:, :, 0]**2 + direction_field_resized[:, :, 1]**2)
        mask = magnitude > 0.01
        direction_field_resized[mask, 0] /= magnitude[mask]
        direction_field_resized[mask, 1] /= magnitude[mask]
        
        direction_tensor = torch.from_numpy(direction_field_resized).permute(2, 0, 1).float()
        
        direction_mask_resized = cv2.resize(
            direction_mask,
            (self.heatmap_size[1], self.heatmap_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        direction_mask_tensor = torch.from_numpy(direction_mask_resized).float()
        
        offset_map_resized = cv2.resize(
            offset_map,
            (self.heatmap_size[1], self.heatmap_size[0]),
            interpolation=cv2.INTER_NEAREST 
        )
        offset_tensor = torch.from_numpy(offset_map_resized).permute(2, 0, 1).float()
        
        heatmap_tensor = heatmap_tensor.unsqueeze(0)
        
        return image_tensor, heatmap_tensor, direction_tensor, direction_mask_tensor, offset_tensor

def create_dataloader(image_dir, 
                     label_dir, 
                     batch_size=4,
                     image_size=(512, 512),
                     sigma_major=17,
                     sigma_minor=4,
                     num_workers=4,
                     train=True,
                     use_augmentation=True,
                     shuffle=True,
                     heatmap_downsample_factor=1,
                     file_list=None):
    """
    Create a DataLoader for hair follicle detection.
    Args:
        image_dir (str): Directory containing input images
        label_dir (str): Directory containing JSON annotation files
        batch_size (int): Number of samples per batch, default 4
        image_size (tuple): Target image size (height, width), default (512, 512)
        sigma_major (float): Gaussian kernel std dev along major axis, default 17
        sigma_minor (float): Gaussian kernel std dev along minor axis, default 4
        num_workers (int): Number of worker processes for data loading, default 4
        train (bool): Whether this is for training, default True
        use_augmentation (bool): Whether to apply data augmentation, default True
        shuffle (bool): Whether to shuffle the data, default True
        heatmap_downsample_factor (int): Downsampling factor for output maps, default 1
        file_list (list, optional): List of filename stems (without extension) to include in dataset
        
    Returns:
        tuple: A tuple containing:
            - dataloader (DataLoader): PyTorch DataLoader instance
            - dataset (HairFollicleDataset): The underlying dataset instance
    """
    dataset = HairFollicleDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        train=train,
        use_augmentation=use_augmentation,
        heatmap_downsample_factor=heatmap_downsample_factor,
        file_list=file_list
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train 
    )
    
    return dataloader, dataset
