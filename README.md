# From Bounding Boxes to Semantic Orientation: Point-Based Follicle Pose Estimation

## Abstract

Accurate estimation of follicle location and hair growth orientation is essential for the image-guided hair transplantation. Most existing trichoscopic analysis approaches rely on bounding box-based approaches, which only provide coarse spatial localization and infer orientation from region geometry rather than the semantic growth direction of hair. Such representations are flawed in challenging scenarios involving occlusion and hair crossings. To address these issues, we reformulate follicle analysis as a point-based pose estimation problem and propose a Follicular Point–Orientation (FPO) representation, in which each follicle is explicitly modeled by a spatial point and an associated unit orientation vector. This formulation decouples localization from bounding-box geometry and enables a compact, semantically grounded description of follicle pose. Based on FPO, we develop a joint follicle position–orientation estimation framework (FPO-Net) for the joint estimation of follicle position and growth direction in trichoscopic images. Experiments on three trichoscopic datasets demonstrate that the proposed point and orientation formulation consistently outperforms rotated bounding boxes methods in both localization precision and orientation estimation. 

## Overview

![Figure 1](images/fig1.png)
*Comparisons between representations based on bounding boxes and ours.*

## Keywords

- Hair follicle detection
- Hair transplantation
- Hair orientation estimation
- Oriented object detection

## Code Release

### � Coming Soon

The complete source code, including the full model implementation and pretrained weights, will be publicly released after paper acceptance.

### 📦 Currently Available

We provide the **model components** to demonstrate our approach:

- ✅ **Deformable Convolution** - Handles irregular geometric deformations
- ✅ **CBAM Attention** - Channel and spatial attention mechanisms
- ✅ **Elongated Feature Extractor** - Direction-aware asymmetric convolutions
- ✅ **ADA Module** - Adaptive Direction Aware Module with dynamic gating
- ✅ **Output Heads** - Heatmap, Vector, and Offset prediction heads



