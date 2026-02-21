import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import DeformConv2d


class DeformableConv2d(nn.Module):

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
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ElongatedFeatureExtractor(nn.Module):
    def __init__(self, channels, reduction=2):
        super(ElongatedFeatureExtractor, self).__init__()
        
        mid_channels = channels // reduction
        
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_v = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_d = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        feat_h = self.conv_h(x)  
        feat_v = self.conv_v(x)
        feat_d = self.conv_d(x) 
        
        feat_all = torch.cat([feat_h, feat_v, feat_d], dim=1) 
        
        out = self.fusion(feat_all)
        
        return out

class ADA_Module(nn.Module):

    def __init__(self, in_channels, out_channels, use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
        super(ADA_Module, self).__init__()
        
        self.use_deformable = use_deformable
        self.use_attention = use_attention
        self.use_elongated = use_elongated
        self.use_gate = use_gate
        
        if use_deformable:
            self.conv1 = DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        num_branches = 1 + int(use_elongated) + int(use_attention)

        if use_elongated:
            self.elongated_extractor = ElongatedFeatureExtractor(out_channels, reduction=2)
        
        if use_attention:
            self.attention = CBAM(out_channels)
        
        if use_gate:
            self.gate_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(out_channels * num_branches, num_branches, kernel_size=1),
                nn.Softmax(dim=1) 
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        identity = out 
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        features = [out]  
        
        if self.use_elongated:
            elongated_feat = self.elongated_extractor(out)
            features.append(elongated_feat)
        
        if self.use_attention:
            attended_feat = self.attention(out)
            features.append(attended_feat)
        
        if self.use_gate:
            all_feats = torch.cat(features, dim=1)  
            
            gates = self.gate_conv(all_feats) 
            
            fused_feat = 0
            for i, feat in enumerate(features):
                gate_weight = gates[:, i:i+1, :, :]
                fused_feat = fused_feat + gate_weight * feat
        else:
            fused_feat = sum(features) / len(features)
        
        out = fused_feat + identity
        
        return out


DeformableFusionBlock = ADA_Module

class Decoder(nn.Module):

    def __init__(self, backbone_channels, output_channels=256, 
                 use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
        super(Decoder, self).__init__()
        
        c1_ch, c2_ch, c3_ch, c4_ch = backbone_channels
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c4_ch, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(c3_ch, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.merge1 = DeformableFusionBlock(
            output_channels * 2, output_channels,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(c2_ch, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.merge2 = DeformableFusionBlock(
            output_channels * 2, output_channels,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_conv3 = nn.Sequential(
            nn.Conv2d(c1_ch, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.merge3 = DeformableFusionBlock(
            output_channels * 2, output_channels,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate
        )
    
    def forward(self, c1, c2, c3, c4):

        up1 = self.up1(c4) 
        skip1 = self.skip_conv1(c3)  
        x = torch.cat([up1, skip1], dim=1)  
        x = self.merge1(x)  
        
        up2 = self.up2(x)  
        skip2 = self.skip_conv2(c2)  
        x = torch.cat([up2, skip2], dim=1) 
        x = self.merge2(x)  
        
        up3 = self.up3(x) 
        skip3 = self.skip_conv3(c1) 
        x = torch.cat([up3, skip3], dim=1) 
        x = self.merge3(x)
        
        return x

class HeatmapHead(nn.Module):
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

class HairFollicleDetector(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, pretrained_path=None,
                 use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
        super(HairFollicleDetector, self).__init__()
        
        self.use_deformable = use_deformable
        self.use_attention = use_attention
        self.use_elongated = use_elongated
        self.use_gate = use_gate
        
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=False)
            deconv_in_channels = 2048 
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False)
            deconv_in_channels = 512  
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=False)
            deconv_in_channels = 2048 
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if pretrained:
            if pretrained_path is not None:
                print(f"Loading pretrained weights from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                resnet.load_state_dict(state_dict, strict=False)
                print("Pretrained weights loaded successfully!")
            else:
                print(f"Downloading pretrained {backbone} weights...")
                if backbone == 'resnet50':
                    pretrained_resnet = models.resnet50(pretrained=True)
                elif backbone == 'resnet34':
                    pretrained_resnet = models.resnet34(pretrained=True)
                elif backbone == 'resnet101':
                    pretrained_resnet = models.resnet101(pretrained=True)
                resnet.load_state_dict(pretrained_resnet.state_dict())
                print("Pretrained weights loaded!")
        else:
            print("Training from scratch (no pretrained weights)")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
        
        if backbone == 'resnet50' or backbone == 'resnet101':
            backbone_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            backbone_channels = [64, 128, 256, 512]
        
        self.decoder = Decoder(
            backbone_channels=backbone_channels,
            output_channels=256,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate
        )
        
        print(f"Model initialized with: Deformable Conv={use_deformable}, Attention={use_attention}, Elongated Conv={use_elongated}, Gate={use_gate}")
        
        self.heatmap_head = HeatmapHead(in_channels=256, num_keypoints=1)
        self.vector_head = VectorHead(in_channels=256)
        self.offset_head = OffsetHead(in_channels=256) 
        
    def forward(self, x):
        x = self.conv1(x)     
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   
        
        c1 = self.layer1(x)    
        c2 = self.layer2(c1)  
        c3 = self.layer3(c2)   
        c4 = self.layer4(c3)   
        
        x = self.decoder(c1, c2, c3, c4) 
        
        heatmap = self.heatmap_head(x)  
        vectors = self.vector_head(x)   
        offset = self.offset_head(x)    
        
        return heatmap, vectors, offset
    
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return c1, c2, c3, c4

def build_model(backbone='resnet50', pretrained=True, pretrained_path=None,
                use_deformable=True, use_attention=True, use_elongated=True, use_gate=True):
    model = HairFollicleDetector(
        backbone=backbone,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        use_deformable=use_deformable,
        use_attention=use_attention,
        use_elongated=use_elongated,
        use_gate=use_gate
    )
    return model

