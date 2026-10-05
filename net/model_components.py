"""
FPO-Net Model Components

Building blocks used by the complete FPO-Net architecture in model.py.

Components included:
- Deformable Convolution
- CBAM Attention (Channel + Spatial)
- Elongated Feature Extractor (Direction-aware convolutions)
- ADA Module (Adaptive Direction Aware Module)
- Multi-scale Decoder structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution wrapper for learning spatial offsets.
    Used to handle irregular geometric deformations of follicles.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    Re-weights feature channels based on their importance.
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    Re-weights spatial locations based on their importance.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention for feature refinement.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ElongatedFeatureExtractor(nn.Module):
    """
    Direction-aware feature extractor using asymmetric convolutions.
    Captures horizontal and vertical elongated structures (hair shafts).
    """
    def __init__(self, channels, reduction=2):
        super(ElongatedFeatureExtractor, self).__init__()
        
        mid_channels = channels // reduction
        
        # Horizontal elongated convolution (1×7)
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Vertical elongated convolution (7×1)
        self.conv_v = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Diagonal/standard convolution (3×3)
        self.conv_d = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        feat_h = self.conv_h(x)  # Horizontal features
        feat_v = self.conv_v(x)  # Vertical features
        feat_d = self.conv_d(x)  # Diagonal features
        
        feat_all = torch.cat([feat_h, feat_v, feat_d], dim=1) 
        out = self.fusion(feat_all)
        
        return out


class ADA_Module(nn.Module):
    """
    Adaptive Direction Aware Module.
    
    Core component of FPO-Net that integrates three parallel branches:
    1. Deformable convolution branch: handles irregular geometric deformations
    2. Elongated feature extractor: captures directional structures
    3. Attention branch (CBAM): re-weights features spatially and channel-wise
    
    A dynamic gating mechanism adaptively fuses the three branches.
    """
    def __init__(self, in_channels, out_channels, 
                 use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
        super(ADA_Module, self).__init__()
        
        self.use_deformable = use_deformable
        self.use_attention = use_attention
        self.use_elongated = use_elongated
        self.use_gate = use_gate
        
        # First convolution (deformable or standard)
        if use_deformable:
            self.conv1 = DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Count number of branches for gating
        num_branches = 1 + int(use_elongated) + int(use_attention)

        # Branch 1: Elongated feature extractor (direction-aware)
        if use_elongated:
            self.elongated_extractor = ElongatedFeatureExtractor(out_channels, reduction=2)
        
        # Branch 2: Attention module
        if use_attention:
            self.attention = CBAM(out_channels)
        
        # Dynamic gating mechanism
        if use_gate:
            self.gate_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global pooling
                nn.Conv2d(out_channels * num_branches, num_branches, kernel_size=1),
                nn.Softmax(dim=1)  # Normalize weights
            )
    
    def forward(self, x):
        # Initial convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        identity = out  # For residual connection
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Collect features from different branches
        features = [out]  # Base branch
        
        if self.use_elongated:
            elongated_feat = self.elongated_extractor(out)
            features.append(elongated_feat)
        
        if self.use_attention:
            attended_feat = self.attention(out)
            features.append(attended_feat)
        
        # Adaptive fusion using gating mechanism
        if self.use_gate:
            all_feats = torch.cat(features, dim=1)  
            gates = self.gate_conv(all_feats)  # (B, num_branches, 1, 1)
            
            fused_feat = 0
            for i, feat in enumerate(features):
                gate_weight = gates[:, i:i+1, :, :]
                fused_feat = fused_feat + gate_weight * feat
        else:
            # Simple averaging if no gate
            fused_feat = sum(features) / len(features)
        
        # Residual connection
        out = fused_feat + identity
        
        return out


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling, skip connection, and ADA fusion.
    """
    def __init__(self, in_channels, skip_channels, out_channels,
                 use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
        super(DecoderBlock, self).__init__()
        
        # Upsampling path
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion with ADA module
        self.merge = ADA_Module(
            out_channels * 2, out_channels,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: Upsampled features from deeper layer
            skip: Skip connection features from encoder
        """
        x_up = self.upsample(x)
        skip_processed = self.skip_conv(skip)
        merged = torch.cat([x_up, skip_processed], dim=1)
        out = self.merge(merged)
        return out


class HeatmapHead(nn.Module):
    """
    Output head for follicle location heatmap prediction.
    Outputs a probability map indicating follicle positions.
    """
    def __init__(self, in_channels=256, num_keypoints=1):
        super(HeatmapHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.head(x)


class VectorHead(nn.Module):
    """
    Output head for growth direction vector prediction.
    Outputs 2-channel direction vectors (vx, vy).
    """
    def __init__(self, in_channels=256):
        super(VectorHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )
        
    def forward(self, x):
        vectors = self.head(x)
        return vectors


class OffsetHead(nn.Module):
    """
    Output head for sub-pixel offset prediction.
    Refines keypoint localization beyond discrete grid positions.
    """
    def __init__(self, in_channels=256):
        super(OffsetHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1), 
        )
        
    def forward(self, x):
        offset = self.head(x)
        return offset


# Note: The complete FPO-Net model (HairFollicleDetector) integrates:
# - ResNet backbone for feature extraction
# - Multi-scale decoder with ADA modules
# - Three output heads (heatmap, vector, offset)
# See model.py for the complete architecture.
