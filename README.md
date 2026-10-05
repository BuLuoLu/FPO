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

## Evaluation

Evaluate existing prediction and ground-truth JSON files in the FPO format;
this tool does not run model inference and only requires NumPy.
For image ID `sample`, use `sample.json` for ground truth and
`sample_result.json` (or `sample.json`) for predictions, at the same original-image
pixel scale. Add `confidence` or `score` to each prediction (default: 1.0).
For images with no predictions, provide a JSON file with `"detections": []`.

```bash
python tools/evaluate.py \
  --pred_dir /path/to/predictions \
  --gt_dir /path/to/annotations \
  --image_names sample \
  --output_dir output/evaluation
```

Results include localization error (MLE), directed angular error (MAE), joint F1,
and joint AP/mAP, saved to `output/evaluation/evaluation_results.json`.
