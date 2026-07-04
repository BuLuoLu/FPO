# FPO-Net: Joint Follicular Opening Localization and Growth Direction Estimation in Trichoscopic Images

## Abstract

Accurate localization of follicular openings and estimation of local hair growth directions are important for the analysis of trichoscopic images. Existing approaches commonly formulate follicle analysis using horizontal or rotated bounding boxes. However, these representations provide only coarse opening localization, while rotated boxes encode an undirected geometric axis rather than the semantic direction of hair growth. To better align the output representation with the task, we formulate follicle analysis as joint point and directed orientation estimation and introduce the Follicular Point-Orientation (FPO) representation. Each follicle is represented by an opening point and an associated unit direction vector, decoupling follicular localization and local orientation from the spatial extent of the hair shaft. Based on FPO, we develop FPO-Net, a joint estimation framework with heatmap, offset, and orientation heads, together with an Adaptive Direction-Aware module for feature refinement. Experiments on three datasets show that the complete FPO-Net achieves lower localization, axial-orientation, and directed-orientation errors, as well as higher joint detection scores, than the evaluated rotated-box-based detectors.

## Overview

![Figure 1](images/fig1.png)
*Comparisons between representations based on bounding boxes and ours.*

## Keywords

- Hair follicle detection
- Hair transplantation
- Hair orientation estimation
- Oriented object detection

## Code

### � Coming Soon

The complete source code, including the full model implementation and pretrained weights, will be publicly released after paper acceptance.

### 📦 Currently Available

We provide the **model components** to demonstrate our approach:

- ✅ **ADA Module** - Adaptive Direction Aware Module with dynamic gating
- ✅ **Output Heads** - Heatmap, Vector, and Offset prediction heads

## FPO Representation

### Overview

The FPO representation describes each follicle instance using a **point-vector** format, where:

- **Point** (`center`): The 2D coordinates of the follicle opening
- **Vector** (`direction`): The semantic growth direction as a unit vector

### Annotation Example

<table>
  <tr>
    <td align="center">
      <img src="annotations/4.jpg" width="400" alt="Original Image"/><br/>
      <b>Original Trichoscopic Image</b>
    </td>
    <td align="center">
      <img src="annotations/result.png" width="400" alt="Annotated Result"/><br/>
      <b>FPO Annotation Visualization</b>
    </td>
  </tr>
</table>

*Left: Original trichoscopic image. Right: FPO annotations with follicle locations (circles) and growth directions (arrows). See [`annotations/4.json`](annotations/4.json) for the complete annotation data.*

### JSON Structure

Each annotation file contains:

```json
{
  "num_detections": 43,           // Total number of follicles in the image
  "image_size": [1024, 1280],     // [height, width] of the image
  "detections": [                 // Array of follicle instances
    {
      "center": [826.24, 520.96], // Follicle location (x, y) in pixels
      "direction": {
        "cos": 0.9676,            // x-component of unit vector (vx)
        "sin": -0.2525,           // y-component of unit vector (vy)
        "angle_rad": -0.2553,     // Angle in radians
        "angle_deg": -14.63       // Angle in degrees
      }
    },
    // ... more detections
  ]
}
```

### Field Descriptions

| Field              | Type           | Description                                                     |
| ------------------ | -------------- | --------------------------------------------------------------- |
| `num_detections` | int            | Total number of follicle instances                              |
| `image_size`     | [int, int]     | Image dimensions [height, width]                                |
| `center`         | [float, float] | Follicle location [x, y] in pixel coordinates                   |
| `cos`            | float          | x-component of growth direction (vx), normalized to unit length |
| `sin`            | float          | y-component of growth direction (vy), normalized to unit length |
| `angle_rad`      | float          | Growth angle in radians, range: [-π, π]                       |
| `angle_deg`      | float          | Growth angle in degrees, range: [-180°, 180°]                 |
