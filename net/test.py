import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from pathlib import Path
import argparse
import yaml
import json

from model import build_model
from utils import extract_keypoints_with_offset

class HairFollicleInferenceWithOffset:

    def __init__(self, checkpoint_path, config_path='config.yaml', device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.downsample_factor = self.config['data']['heatmap_downsample_factor']
        
        print(f"Model loaded successfully! (with Offset subpixel localization)")
        print(f"Device: {self.device}")
        print(f"Downsample factor: {self.downsample_factor}")
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        model = build_model(
            backbone=self.config['model']['backbone'],
            pretrained=False,
            use_deformable=self.config['model'].get('use_deformable', True),
            use_attention=self.config['model'].get('use_attention', True),
            use_elongated=self.config['model'].get('use_elongated', True)
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path, target_size=(1024, 1024)):
        # 读取原始图像
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        orig_h, orig_w = original_image.shape[:2]
        
        # Resize到目标尺寸
        image_resized = cv2.resize(original_image, (target_size[1], target_size[0]))
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # 标准化（ImageNet）
        image_norm = (image_norm - self.mean) / self.std
        
        # 转换为tensor [1, 3, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # 计算缩放因子
        scale_h = orig_h / target_size[0]
        scale_w = orig_w / target_size[1]
        
        return image_tensor, original_image, (scale_h, scale_w)
    
    @torch.no_grad()
    def predict(self, image_path, threshold=0.3, nms_kernel=5):

        image_tensor, original_image, scale_factors = self.preprocess_image(image_path)
    
        pred_heatmap, pred_direction, pred_offset = self.model(image_tensor)
        
        heatmap = pred_heatmap[0, 0].cpu().numpy()  
        direction = pred_direction[0].cpu().numpy() 
        offset = pred_offset[0].cpu().numpy()        
        
        keypoints_feat = extract_keypoints_with_offset(
            heatmap, 
            offset, 
            threshold=threshold, 
            nms_kernel=nms_kernel,
            downsample_ratio=self.downsample_factor
        )
        
        points_original = []
        directions_original = []
        confidences = []
        
        for (x_feat, y_feat, confidence) in keypoints_feat:
            x_1024 = x_feat * self.downsample_factor
            y_1024 = y_feat * self.downsample_factor
            
            x_orig = x_1024 * scale_factors[1]
            y_orig = y_1024 * scale_factors[0]
            
            x_idx = int(np.clip(x_feat, 0, direction.shape[2] - 1))
            y_idx = int(np.clip(y_feat, 0, direction.shape[1] - 1))
            
            cos_theta = direction[0, y_idx, x_idx]
            sin_theta = direction[1, y_idx, x_idx]
            
            points_original.append((x_orig, y_orig))
            directions_original.append((cos_theta, sin_theta))
            confidences.append(confidence)
        
        results = {
            'heatmap': heatmap,
            'direction': direction,
            'offset': offset, 
            'points_original': points_original,
            'directions': directions_original,
            'confidences': confidences,
            'original_image': original_image,
            'scale_factors': scale_factors
        }
        
        return results
    
    def visualize_results(self, results, save_path=None, arrow_length=50):

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        original_image = results['original_image']
        heatmap = results['heatmap']
        direction = results['direction']
        offset = results['offset']
        points_original = results['points_original']
        directions = results['directions']
        
        ax = axes[0, 0]
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title('Original Image', fontsize=14)
        ax.axis('off')
        
        ax = axes[0, 1]
        ax.imshow(image_rgb)
        
        orig_h, orig_w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        
        im = ax.imshow(heatmap_resized, alpha=0.6, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'Heatmap Overlay ({len(points_original)} detections)', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[0, 2]
        cos_map = direction[0]
        sin_map = direction[1]
        angle_map = np.arctan2(sin_map, cos_map)
        angle_deg = (np.degrees(angle_map) + 360) % 360
        
        hsv = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle_deg / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (heatmap * 255).astype(np.uint8)
        
        direction_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        direction_vis_resized = cv2.resize(direction_vis, (orig_w, orig_h))
        
        ax.imshow(direction_vis_resized)
        ax.set_title('Direction Field (HSV)', fontsize=14)
        ax.axis('off')
        
        ax = axes[1, 0]
        offset_x = offset[0]
        vmin_x = np.percentile(offset_x, 1)  
        vmax_x = np.percentile(offset_x, 99) 
        if vmax_x - vmin_x < 0.01:
            vmin_x, vmax_x = -0.5, 0.5
        
        im = ax.imshow(offset_x, cmap='coolwarm', vmin=vmin_x, vmax=vmax_x)
        ax.set_title(f'Offset X (range: [{vmin_x:.3f}, {vmax_x:.3f}])', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, 1]
        offset_y = offset[1]
        vmin_y = np.percentile(offset_y, 1)
        vmax_y = np.percentile(offset_y, 99)
        if vmax_y - vmin_y < 0.01:
            vmin_y, vmax_y = -0.5, 0.5
        
        im = ax.imshow(offset_y, cmap='coolwarm', vmin=vmin_y, vmax=vmax_y)
        ax.set_title(f'Offset Y (range: [{vmin_y:.3f}, {vmax_y:.3f}])', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        print(f"  Offset X: mean={offset_x.mean():.4f}, std={offset_x.std():.4f}, range=[{offset_x.min():.4f}, {offset_x.max():.4f}]")
        print(f"  Offset Y: mean={offset_y.mean():.4f}, std={offset_y.std():.4f}, range=[{offset_y.min():.4f}, {offset_y.max():.4f}]")
        
        ax = axes[1, 2]
        ax.imshow(image_rgb)
        
        for (x, y), (cos_theta, sin_theta) in zip(points_original, directions):
            ax.plot(x, y, 'yo', markersize=8, markeredgewidth=2, markeredgecolor='red')
            
            dx = cos_theta * arrow_length
            dy = sin_theta * arrow_length
            
            ax.arrow(x, y, dx, dy, 
                    head_width=15, head_length=15,
                    fc='red', ec='red', linewidth=2)
        
        ax.set_title(f'Detection Results (With Offset): {len(points_original)} follicles', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.close()
    
    def save_detection_results(self, results, save_path, arrow_length=50):

        image = results['original_image'].copy()
        points = results['points_original']
        directions = results['directions']
        
        for (x, y), (cos_theta, sin_theta) in zip(points, directions):
            x, y = int(x), int(y)

            cv2.circle(image, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(image, (x, y), 8, (0, 0, 255), 2)
            
            end_x = int(x + cos_theta * arrow_length)
            end_y = int(y + sin_theta * arrow_length)
            
            cv2.arrowedLine(image, (x, y), (end_x, end_y), 
                          (0, 0, 255), 3, tipLength=0.3)
        
        cv2.imwrite(str(save_path), image)
        print(f"Detection result saved: {save_path}")
    
    def export_to_json(self, results, save_path):

        detections = []
        for (x, y), (cos_theta, sin_theta), confidence in zip(
            results['points_original'], 
            results['directions'],
            results['confidences']
        ):
            angle_rad = np.arctan2(sin_theta, cos_theta)
            angle_deg = np.degrees(angle_rad)
            
            detections.append({
                'center': [float(x), float(y)],
                'confidence': float(confidence),
                'direction': {
                    'cos': float(cos_theta),
                    'sin': float(sin_theta),
                    'angle_rad': float(angle_rad),
                    'angle_deg': float(angle_deg)
                }
            })
        
        output = {
            'num_detections': len(detections),
            'image_size': results['original_image'].shape[:2],
            'detections': detections,
            'use_offset': True 
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"JSON result saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Hair Follicle Detection with Offset (Subpixel Localization)')
    parser.add_argument('--image', type=str, required=False, 
                       help='Input image path')
    parser.add_argument('--val_list', type=str, required=False,
                       help='Val list file (one image name per line)')
    parser.add_argument('--image_dir', type=str, required=False,
                       help='Image directory (used with val_list)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, default='output/inference_with_offset',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.63,
                       help='Heatmap detection threshold (0-1, default 0.63)')
    parser.add_argument('--nms_kernel', type=int, default=5,
                       help='NMS kernel size (odd number, default 5)')
    parser.add_argument('--arrow_length', type=int, default=50,
                       help='Direction arrow length (pixels)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_img_dir = output_dir / 'result_img'
    result_json_dir = output_dir / 'result_json'
    result_img_dir.mkdir(parents=True, exist_ok=True)
    result_json_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Hair Follicle Detection with Offset (Subpixel Localization)")
    print("="*60 + "\n")
    

    inferencer = HairFollicleInferenceWithOffset(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    if args.val_list:
        if not args.image_dir:
            print("Error: --image_dir is required when using --val_list!")
            return
        
        with open(args.val_list, 'r', encoding='utf-8') as f:
            img_names = [line.strip() for line in f if line.strip()]
        
        print(f"Batch inference for {len(img_names)} images...")
        
        for idx, img_name in enumerate(img_names, 1):
            if not Path(img_name).suffix:
                img_name_full = img_name + '.jpg'
            else:
                img_name_full = img_name
            
            img_path = Path(args.image_dir) / img_name_full
            
            if not img_path.exists():
                print(f"[{idx}/{len(img_names)}] Skip non-existent image: {img_path}")
                continue
            
            print(f"\n[{idx}/{len(img_names)}] Processing: {img_path.name}")
            
            try:
                results = inferencer.predict(
                    image_path=str(img_path),
                    threshold=args.threshold,
                    nms_kernel=args.nms_kernel
                )
                
                print(f"  Detected {len(results['points_original'])} follicles")
                
                image_stem = Path(img_path).stem
                
                vis_path = result_img_dir / f"{image_stem}_visualization.png"
                inferencer.visualize_results(
                    results,
                    save_path=vis_path,
                    arrow_length=args.arrow_length
                )
                
                result_path = result_img_dir / f"{image_stem}_result.jpg"
                inferencer.save_detection_results(
                    results,
                    save_path=result_path,
                    arrow_length=args.arrow_length
                )
                
                json_path = result_json_dir / f"{image_stem}_result.json"
                inferencer.export_to_json(results, save_path=json_path)
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        print("\n" + "="*60)
        print("Batch inference completed!")
        print(f"Output directory: {output_dir}")
        print(f"  - Image results: {result_img_dir}")
        print(f"  - JSON results: {result_json_dir}")
        print("="*60 + "\n")
    
    elif args.image:
        print(f"Processing: {args.image}")
        print(f"Parameters: threshold={args.threshold}, nms_kernel={args.nms_kernel}")
        
        results = inferencer.predict(
            image_path=args.image,
            threshold=args.threshold,
            nms_kernel=args.nms_kernel
        )
        
        print(f"Detected {len(results['points_original'])} follicles")
        
        image_stem = Path(args.image).stem
        
        vis_path = result_img_dir / f"{image_stem}_visualization.png"
        inferencer.visualize_results(
            results,
            save_path=vis_path,
            arrow_length=args.arrow_length
        )
        
        result_path = result_img_dir / f"{image_stem}_result.jpg"
        inferencer.save_detection_results(
            results,
            save_path=result_path,
            arrow_length=args.arrow_length
        )
        
        json_path = result_json_dir / f"{image_stem}_result.json"
        inferencer.export_to_json(results, save_path=json_path)
        
        print("\n" + "="*60)
        print("Inference completed!")
        print(f"Output directory: {output_dir}")
        print("="*60 + "\n")
