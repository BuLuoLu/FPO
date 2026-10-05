# FPO-Net: Joint Follicular Opening Localization and Growth Direction Estimation in Trichoscopic Images

FPO-Net jointly localizes follicular openings and estimates hair growth
directions using a point-vector representation.

![FPO-Net overview](images/fig1.png)

## Installation

Run commands from the repository root. Install compatible PyTorch/torchvision
versions with support for `torchvision.ops.DeformConv2d`.

```bash
pip install -r requirements.txt
```

## Model

```python
from net import build_model

model = build_model(backbone="resnet50", pretrained=False)
```

## Annotations

Each follicle is represented by a pixel location `center: [x, y]` and a unit
growth vector `direction: {cos, sin, angle_rad, angle_deg}`. Coordinates use
the original image frame, with x increasing rightward and y downward.

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

See [`annotations/4.json`](annotations/4.json) for the example and
[`annotations/README.md`](annotations/README.md) for the format and annotation
protocol. To visualize follicle locations and growth directions:

```bash
python tools/visualize_annotation.py \
  --image annotations/4.jpg \
  --annotation annotations/4.json \
  --output output/annotation.png
```

## OBB Annotation Conversion

Convert LabelMe triangle annotations (follicular opening first) to four-corner
OBB polygons.

```bash
python tools/convert_triangle_to_rotated_bbox.py \
  --input_dir /path/to/triangle_annotations \
  --output_dir output/obb \
  --extension_ratio 0.0
```

## Evaluation

Evaluate FPO predictions using MLE, directed MAE, joint F1, and joint AP/mAP.
For image `sample`, use `sample_result.json` for predictions and `sample.json`
for ground truth.

```bash
python tools/evaluate.py \
  --pred_dir /path/to/predictions \
  --gt_dir /path/to/annotations \
  --image_names sample \
  --output_dir output/evaluation
```

Results are saved to `output/evaluation/evaluation_results.json`.
