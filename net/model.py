"""FPO-Net: a ResNet encoder, an ADA decoder, and three prediction heads."""

import torch
from torch import nn
from torchvision import models

if __package__:
    from .model_components import ADA_Module, HeatmapHead, OffsetHead, VectorHead
else:
    from model_components import ADA_Module, HeatmapHead, OffsetHead, VectorHead


class UNetDecoder(nn.Module):
    """Fuse encoder features at strides 16, 8, and 4 using ADA modules."""

    def __init__(self, backbone_channels, output_channels=256,
                 use_deformable=True, use_attention=True,
                 use_elongated=True, use_gate=True):
        super().__init__()
        c1, c2, c3, c4 = backbone_channels
        # Keep stage names stable when loading existing model state dictionaries.
        for stage, in_channels, skip_channels in (
            (1, c4, c3), (2, output_channels, c2), (3, output_channels, c1)
        ):
            setattr(self, f"up{stage}", nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ))
            setattr(self, f"skip_conv{stage}", nn.Sequential(
                nn.Conv2d(skip_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ))
            setattr(self, f"merge{stage}", ADA_Module(
                output_channels * 2, output_channels,
                use_deformable=use_deformable,
                use_attention=use_attention,
                use_elongated=use_elongated,
                use_gate=use_gate,
            ))

    def forward(self, c1, c2, c3, c4):
        x = c4
        for stage, skip in enumerate((c3, c2, c1), start=1):
            up = getattr(self, f"up{stage}")(x)
            skip = getattr(self, f"skip_conv{stage}")(skip)
            x = getattr(self, f"merge{stage}")(torch.cat((up, skip), dim=1))
        return x


class HairFollicleDetector(nn.Module):
    """Predict an opening heatmap, direction vectors, and subpixel offsets.

    Inputs have shape ``[B, 3, H, W]``; H and W must be multiples of 32.
    Outputs, in order, have shapes ``[B, 1, H/4, W/4]``,
    ``[B, 2, H/4, W/4]``, and ``[B, 2, H/4, W/4]``.
    Heatmaps use sigmoid; vectors and offsets are raw regression outputs.

    ``pretrained`` refers only to ImageNet encoder weights, not FPO-Net
    task weights. Downloads are disabled by default. ``pretrained_path``
    can supply a local ResNet state dictionary when ``pretrained=True``.
    """

    def __init__(self, backbone="resnet50", pretrained=False, pretrained_path=None,
                 use_deformable=True, use_attention=True,
                 use_elongated=True, use_gate=True):
        super().__init__()
        backbones = {
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1,
                         [64, 128, 256, 512]),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1,
                         [256, 512, 1024, 2048]),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1,
                          [256, 512, 1024, 2048]),
        }
        if backbone not in backbones:
            raise ValueError(f"Unsupported backbone: {backbone}")
        constructor, imagenet_weights, channels = backbones[backbone]
        weights = imagenet_weights if pretrained and pretrained_path is None else None
        resnet = constructor(weights=weights)
        if pretrained and pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            resnet.load_state_dict(state_dict, strict=False)

        self.use_deformable = use_deformable
        self.use_attention = use_attention
        self.use_elongated = use_elongated
        self.use_gate = use_gate
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.decoder = UNetDecoder(
            backbone_channels=channels,
            output_channels=256,
            use_deformable=use_deformable,
            use_attention=use_attention,
            use_elongated=use_elongated,
            use_gate=use_gate,
        )
        self.heatmap_head = HeatmapHead(in_channels=256, num_keypoints=1)
        self.vector_head = VectorHead(in_channels=256)
        self.offset_head = OffsetHead(in_channels=256)

    def extract_features(self, x):
        """Return encoder feature maps at strides 4, 8, 16, and 32."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

    def forward(self, x):
        features = self.decoder(*self.extract_features(x))
        return (self.heatmap_head(features),
                self.vector_head(features),
                self.offset_head(features))


def build_model(backbone="resnet50", pretrained=False, pretrained_path=None,
                use_deformable=True, use_attention=True,
                use_elongated=True, use_gate=True):
    """Construct FPO-Net without a training or inference command-line entry."""
    return HairFollicleDetector(
        backbone=backbone,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        use_deformable=use_deformable,
        use_attention=use_attention,
        use_elongated=use_elongated,
        use_gate=use_gate,
    )
