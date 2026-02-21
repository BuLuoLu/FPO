"""
Loss Functions for Oriented Object Detection

This module implements various loss functions for training the FPO-Net model:
1. FocalHeatmapLoss - Focal loss for heatmap regression
2. DirectionLoss - Smooth L1 loss for direction vector regression
3. UnitVectorLoss - Constraint loss to ensure unit vector magnitude
4. AngleLoss - Angular loss directly constraining angle values
5. OffsetLoss - L1 loss for sub-pixel localization
6. CombinedLoss - Weighted combination of all losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalHeatmapLoss(nn.Module):
    """
    Formula:
        FL(pt) = -α(1-pt)^γ * log(pt)  for positive samples
        FL(pt) = -(1-α)pt^γ * log(1-pt) for negative samples
    
    where:
        - pt is the predicted probability
        - α balances positive/negative samples (default 0.75)
        - γ focuses on hard examples (default 2.0)
    """
    
    def __init__(self, alpha=0.75, beta=2.0):
        """   
        Args:
            alpha (float): Weighting factor for positive samples, default 0.75
            beta (float): Focusing parameter (gamma in the paper), default 2.0
        """
        super(FocalHeatmapLoss, self).__init__()
        self.alpha = alpha   
        self.gamma = beta  
    
    def forward(self, pred, target):
        """  
        Args:
            pred (torch.Tensor): Predicted heatmap [B, 1, H, W], values in [0, 1]
            target (torch.Tensor): Target heatmap [B, 1, H, W], values in [0, 1]
    
        """
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        pos_loss = -self.alpha * torch.pow(1 - pred, self.gamma) * target * torch.log(pred)
        neg_loss = -(1 - self.alpha) * torch.pow(pred, self.gamma) * (1 - target) * torch.log(1 - pred)
        
        loss = pos_loss + neg_loss
        
        if torch.isinf(loss).any():
            print("  ⚠ FocalHeatmapLoss: inf detected, replacing with 10.0")
            loss = torch.where(torch.isinf(loss), torch.tensor(10.0, device=loss.device), loss)
        
        if torch.isnan(loss).any():
            print("  ⚠ FocalHeatmapLoss: nan detected, replacing with 0.0")
            loss = torch.where(torch.isnan(loss), torch.tensor(0.0, device=loss.device), loss)
        
        return loss.mean()

class DirectionLoss(nn.Module):
    """
    Direction Loss using Smooth L1 Loss.

    Note: Although direction is represented as (cos, sin), treating it as
          a numerical regression problem with Smooth L1 works better than
          cosine similarity loss in practice.
    
    Smooth L1 Loss:
        L(x) = 0.5 * x^2           if |x| < 1
        L(x) = |x| - 0.5           otherwise
    """
    
    def __init__(self):
        super(DirectionLoss, self).__init__()
    
    def forward(self, pred_direction, target_direction, direction_mask):
        """
        Args:
            pred_direction (torch.Tensor): Predicted direction field [B, 2, H, W] 
                                          where channel 0 is cos(θ), channel 1 is sin(θ)
            target_direction (torch.Tensor): Target direction field [B, 2, H, W]
            direction_mask (torch.Tensor): Valid region mask [B, H, W], binary values

        """
        device = pred_direction.device
        
        num = direction_mask.float().sum()
        
        if num == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mask = direction_mask.unsqueeze(1).expand_as(pred_direction).float()
        pred_direction = pred_direction * mask
        target_direction = target_direction * mask
        

        regr_loss = F.smooth_l1_loss(pred_direction, target_direction, reduction='sum')
        # regr_loss = F.l1_loss(pred_direction, target_direction, reduction='sum') 
        regr_loss = regr_loss / num
        
        return regr_loss

class UnitVectorLoss(nn.Module):
    """
    Ensures predicted direction vectors satisfy the unit vector constraint:
    cos²(θ) + sin²(θ) = 1, or equivalently ||v|| = 1.
    
    Mathematical Principle:
        - For unit vector: ||v|| = sqrt(cos²θ + sin²θ) = 1
        - Loss function: L = Smooth_L1(||v|| - 1)
    """
    
    def __init__(self):
        super(UnitVectorLoss, self).__init__()
    
    def forward(self, pred_direction, direction_mask):
        """
        Args:
            pred_direction (torch.Tensor): Predicted direction field [B, 2, H, W]
                                          where channel 0 is cos(θ), channel 1 is sin(θ)
            direction_mask (torch.Tensor): Valid region mask [B, H, W], binary values
        """
        device = pred_direction.device
        
        num = direction_mask.float().sum()
        
        if num == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        magnitude = torch.sqrt(
            pred_direction[:, 0]**2 + pred_direction[:, 1]**2 + 1e-8 
        )
        
        diff = magnitude - 1.0
        
        diff = diff * direction_mask
        
        unit_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='sum')
        # unit_loss = F.l1_loss(diff, torch.zeros_like(diff), reduction='sum')
        return unit_loss / num

class AngleLoss(nn.Module):
    """
    Mathematical Principle:
        - Compute angle from (cos, sin): θ = atan2(sin, cos) ∈ [-π, π]
        - Angle difference considers periodicity: Δθ = ((θ_pred - θ_gt + π) mod 2π) - π
        - Loss function: L = Smooth_L1(Δθ)
    """
    
    def __init__(self):
        super(AngleLoss, self).__init__()
    
    def forward(self, pred_direction, target_direction, direction_mask):
        """
        Args:
            pred_direction (torch.Tensor): Predicted direction field [B, 2, H, W]
                                          where channel 0 is cos(θ), channel 1 is sin(θ)
            target_direction (torch.Tensor): Target direction field [B, 2, H, W]
            direction_mask (torch.Tensor): Valid region mask [B, H, W], binary values

        """
        device = pred_direction.device
        
        num = direction_mask.float().sum()
        
        if num == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        pred_angle = torch.atan2(pred_direction[:, 1], pred_direction[:, 0])
        target_angle = torch.atan2(target_direction[:, 1], target_direction[:, 0])
        
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        angle_diff = angle_diff * direction_mask
        
        angle_loss = F.smooth_l1_loss(angle_diff, torch.zeros_like(angle_diff), reduction='sum')
        
        return angle_loss / num

class OffsetLoss(nn.Module):
    """
    Computes L1 loss for offset predictions, which encode the quantization error
    when mapping continuous coordinates to discrete feature map grids.
    """
    
    def __init__(self):
        super(OffsetLoss, self).__init__()
    
    def forward(self, pred_offset, target_offset, mask):
        """
        Args:
            pred_offset (torch.Tensor): Predicted offset map [B, 2, H, W]
                                       where channel 0 is x-offset, channel 1 is y-offset
            target_offset (torch.Tensor): Target offset map [B, 2, H, W]
            mask (torch.Tensor): Mask indicating object center locations [B, H, W] or [B, 1, H, W]
                                Usually the heatmap is used as mask
        """
        device = pred_offset.device
        
        if mask.dim() == 3: 
            mask = mask.unsqueeze(1)
        
        mask = mask.expand_as(pred_offset)
        
        num_pos = mask.sum() / 2 + 1e-4 
        
        if num_pos < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = F.l1_loss(
            pred_offset * mask, 
            target_offset * mask, 
            reduction='sum'
        )
        
        loss = loss / num_pos
        
        return loss

class CombinedLoss(nn.Module):
    """
    The combination provides:
    - Accurate object center localization (heatmap + offset)
    - Precise orientation estimation (direction + unit + angle)
    - Stable training with multiple complementary supervisions
    """
    
    def __init__(self, 
                 heatmap_weight=1.0,
                 direction_weight=0.1,
                 unit_weight=0.1,
                 angle_weight=0.1,
                 offset_weight=0.1,
                 focal_alpha=0.75,
                 focal_beta=2.0):
        """
        Args:
            heatmap_weight (float): Weight for heatmap loss, default 1.0
            direction_weight (float): Weight for direction regression loss, default 0.1
            unit_weight (float): Weight for unit vector constraint, default 0.1
            angle_weight (float): Weight for angle constraint, default 0.1
            offset_weight (float): Weight for offset loss, default 0.1
            focal_alpha (float): Alpha parameter for focal loss, default 0.75
            focal_beta (float): Beta (gamma) parameter for focal loss, default 2.0
        """
        super(CombinedLoss, self).__init__()
        
        self.heatmap_weight = heatmap_weight
        self.direction_weight = direction_weight
        self.unit_weight = unit_weight
        self.angle_weight = angle_weight
        self.offset_weight = offset_weight
        
        self.heatmap_loss = FocalHeatmapLoss(alpha=focal_alpha, beta=focal_beta)
        self.direction_loss = DirectionLoss()
        self.unit_loss = UnitVectorLoss()
        self.angle_loss = AngleLoss()
        self.offset_loss = OffsetLoss()
    
    def forward(self, pred_heatmap, pred_direction, pred_offset,
                target_heatmap, target_direction, target_offset, direction_mask):
        """
        Args:
            pred_heatmap (torch.Tensor): Predicted heatmap [B, 1, H, W], after sigmoid
            pred_direction (torch.Tensor): Predicted direction field [B, 2, H, W]
            pred_offset (torch.Tensor): Predicted offset map [B, 2, H, W]
            target_heatmap (torch.Tensor): Target heatmap [B, 1, H, W]
            target_direction (torch.Tensor): Target direction field [B, 2, H, W]
            target_offset (torch.Tensor): Target offset map [B, 2, H, W]
            direction_mask (torch.Tensor): Valid region mask for direction [B, H, W]

        """
        loss_heatmap = self.heatmap_loss(pred_heatmap, target_heatmap)
        loss_direction = self.direction_loss(pred_direction, target_direction, direction_mask)
        loss_unit = self.unit_loss(pred_direction, direction_mask)
        loss_angle = self.angle_loss(pred_direction, target_direction, direction_mask)
        loss_offset = self.offset_loss(pred_offset, target_offset, target_heatmap) 
        
        total_loss = (self.heatmap_weight * loss_heatmap + 
                     self.direction_weight * loss_direction +
                     self.unit_weight * loss_unit +
                     self.angle_weight * loss_angle +
                     self.offset_weight * loss_offset)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'heatmap_loss': loss_heatmap.item(),
            'direction_loss': loss_direction.item(),
            'unit_loss': loss_unit.item(),
            'angle_loss': loss_angle.item(),
            'offset_loss': loss_offset.item()
        }
        
        return total_loss, loss_dict

# ============== Factory Function ==============
def build_loss(loss_config=None):
    """
    Create a loss function instance with specified configuration.
    """
    if loss_config is None:
        loss_config = {}
    
    heatmap_weight = loss_config.get('heatmap_weight', 1.0)
    direction_weight = loss_config.get('direction_weight', 0.1)
    unit_weight = loss_config.get('unit_weight', 0.1)
    angle_weight = loss_config.get('angle_weight', 0.1)
    offset_weight = loss_config.get('offset_weight', 0.1)
    focal_alpha = loss_config.get('focal_alpha', 0.75)
    focal_beta = loss_config.get('focal_beta', 2)
    
    criterion = CombinedLoss(
        heatmap_weight=heatmap_weight,
        direction_weight=direction_weight,
        unit_weight=unit_weight,
        angle_weight=angle_weight,
        offset_weight=offset_weight,
        focal_alpha=focal_alpha,
        focal_beta=focal_beta
    )
    
    return criterion
