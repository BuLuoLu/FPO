# FPO Annotations

This directory contains example annotations in the FPO (Follicular Point-Orientation) format.

## Files

- `4.json` - Example annotation file with 43 follicle instances
- `4.jpg` - Corresponding trichoscopic image
- `result.png` - Visualization of annotations on the image

## Annotation Format

Each annotation file is a JSON containing follicle locations and growth directions.

### Structure

```text
{
  "num_detections": <number of follicles>,
  "image_size": [height, width],
  "detections": [
    {
      "center": [x, y],
      "direction": {
        "cos": vx,
        "sin": vy,
        "angle_rad": θ_rad,
        "angle_deg": θ_deg
      }
    },
    ...
  ]
}
```

## Annotation Protocol

### Point Annotation (`center`)
- Mark the **anatomical opening** of the hair shaft
- Coordinates are in pixel space: [x, y]
- For multiple hairs from one follicle: annotate each visible shaft separately
- For crossing hairs: treat each visible shaft as a distinct instance

### Direction Annotation (`direction`)
- Determined by a secondary point on a **fixed-radius circle** around the center
- The vector points from the follicle opening toward hair growth direction
- Automatically normalized to unit length (cos² + sin² = 1)
- Represents **semantic growth orientation**, not geometric axis

### Advantages over Bounding Boxes

| Aspect | FPO (Ours) | Rotated Bounding Box |
|--------|------------|---------------------|
| Location | Explicit point at follicle opening | Inferred from box center |
| Orientation | Semantic growth direction | Geometric axis (ambiguous) |
| Annotation | 2 points per follicle | 4+ points for box corners |
| Ambiguity | Unambiguous direction | 180° periodicity |
| Crossing hairs | Each instance clear | Overlapping boxes |

## Visualization

Use the provided tool to visualize annotations:

Run from the repository root:

```bash
python tools/visualize_annotation.py \
  --image annotations/4.jpg \
  --annotation annotations/4.json \
  --output output/annotation.png
```

This will overlay:
- Follicle positions as circles
- Growth directions as arrows

## Statistics for Example (4.json)

- **Total follicles**: 43
- **Image size**: 1024 × 1280 pixels

## Creating New Annotations

1. Open image in annotation tool
2. For each follicle:
   - Click on follicle opening → records `center`
   - Click on growth direction point (within fixed radius) → records `direction`
3. Export to JSON format

Only this single example image, its annotation, and the visualization are
included. The full datasets and annotation-authoring software are not part
of this release. `result.png` is an annotation visualization, not a model
prediction or an evaluation report.
