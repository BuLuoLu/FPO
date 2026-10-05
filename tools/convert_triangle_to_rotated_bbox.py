#!/usr/bin/env python3
"""Convert LabelMe triangle annotations to four-corner oriented boxes.

The first triangle point is the follicular opening; the other two points
form the base. The opening-to-base-midpoint axis sets the box orientation.
The base is made perpendicular to that axis while preserving its length.
Only annotation JSON files are written; images and split lists are not needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _validate_extension_ratio(extension_ratio):
    ratio = float(extension_ratio)
    if not np.isfinite(ratio) or ratio <= -1.0:
        raise ValueError("extension_ratio must be finite and greater than -1.")
    return ratio


def triangle_to_rotated_bbox(triangle_points, extension_ratio=0.0):
    """Return four cyclically ordered corners using the original triangle rule.

    Args:
        triangle_points: Three [x, y] points, opening first, then base endpoints.
        extension_ratio: Extend behind the opening by this fraction of the
            opening-to-base-midpoint distance. Box length is (1 + ratio) times
            that distance. Values between -1 and 0 shorten the box.

    Coordinates stay in the original pixel frame and are not clipped to image
    bounds. The returned box remains a LabelMe polygon with four points.
    """
    ratio = _validate_extension_ratio(extension_ratio)
    points = np.asarray(triangle_points, dtype=np.float64)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError("Expected three finite [x, y] triangle points.")

    apex, base_p1, base_p2 = points
    base_center = (base_p1 + base_p2) / 2.0
    height_vector = apex - base_center
    height = np.linalg.norm(height_vector)
    base_vector = base_p2 - base_p1
    base_length = np.linalg.norm(base_vector)
    apex_vector = apex - base_p1
    twice_area = base_vector[0] * apex_vector[1] - base_vector[1] * apex_vector[0]
    if height == 0 or base_length == 0 or twice_area == 0:
        raise ValueError("A triangle must have distinct, non-collinear vertices.")

    height_unit = height_vector / height
    base_unit = np.array([-height_unit[1], height_unit[0]])
    half_base = base_length / 2.0
    extended_apex = apex + height_unit * (height * ratio)
    corners = np.array([
        extended_apex - base_unit * half_base,
        extended_apex + base_unit * half_base,
        base_center + base_unit * half_base,
        base_center - base_unit * half_base,
    ])
    if not np.isfinite(corners).all():
        raise ValueError("Triangle coordinates or extension_ratio are too large.")
    return corners.tolist()


def convert_annotation_file(input_path, output_path, extension_ratio=0.0):
    """Convert triangle shapes, preserve other shapes/metadata, and return a count."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output must differ from the source annotation.")
    ratio = _validate_extension_ratio(extension_ratio)
    with input_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        raise ValueError(
            f"{input_path}: expected LabelMe JSON with a 'shapes' list. "
            "FPO center/direction annotations do not specify box dimensions."
        )

    converted_count = 0
    converted_shapes = []
    for index, shape in enumerate(data["shapes"]):
        if not isinstance(shape, dict):
            raise ValueError(f"{input_path}: shape {index} must be an object.")
        points = shape.get("points", [])
        if shape.get("shape_type") == "polygon":
            if not isinstance(points, list):
                raise ValueError(f"{input_path}: shape {index} points must be a list.")
            if len(points) == 3:
                shape = shape.copy()
                try:
                    shape["points"] = triangle_to_rotated_bbox(points, ratio)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{input_path}: shape {index}: {error}") from error
                converted_count += 1
        converted_shapes.append(shape)

    data["shapes"] = converted_shapes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return converted_count


def resolve_anno_dir(input_dir, anno_dir=None):
    """Locate annotations in a dataset root or accept a direct JSON directory."""
    if anno_dir is not None:
        return Path(anno_dir)
    for name in ("labels", "lables", "annotations"):
        candidate = Path(input_dir) / name
        if candidate.is_dir():
            return candidate
    return Path(input_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe triangles to four-corner OBB polygons (NumPy only)."
    )
    parser.add_argument("--input", "--input_dir", dest="input_path", type=Path, required=True,
                        help="A LabelMe JSON file, annotation directory, or dataset root.")
    parser.add_argument("--output", "--output_dir", dest="output_path", type=Path, required=True,
                        help="Output JSON file for single-file mode; output directory for batch mode.")
    parser.add_argument("--anno_dir", type=Path,
                        help="Explicit annotation directory when the input is a dataset root.")
    parser.add_argument("--extension_ratio", type=float, default=0.0,
                        help="Box length multiplier minus one: 0=no extension, 0.5=1.5x, 1=2x.")
    args = parser.parse_args()

    try:
        _validate_extension_ratio(args.extension_ratio)
        if args.input_path.is_file():
            if args.anno_dir is not None:
                parser.error("--anno_dir is only supported for directory input.")
            if args.output_path.suffix.lower() != ".json":
                parser.error("Single-file output must have a .json extension.")
            pairs = [(args.input_path, args.output_path)]
        elif args.input_path.is_dir():
            anno_dir = resolve_anno_dir(args.input_path, args.anno_dir)
            if not anno_dir.is_dir():
                parser.error(f"Annotation directory not found: {anno_dir}")
            if args.output_path.is_file():
                parser.error("Directory input requires an output directory.")
            json_files = sorted(anno_dir.glob("*.json"))
            if not json_files:
                parser.error(f"No JSON annotations found in {anno_dir}")
            pairs = [(path, args.output_path / path.name) for path in json_files]
        else:
            parser.error(f"Input not found: {args.input_path}")

        source_paths = {source.resolve() for source, _ in pairs}
        if any(target.resolve() in source_paths for _, target in pairs):
            parser.error("Output paths must not overwrite source annotations.")

        total = 0
        for source, target in pairs:
            count = convert_annotation_file(source, target, args.extension_ratio)
            total += count
            print(f"{source.name}: converted {count} triangle(s) -> {target}")
        print(f"Converted {total} triangle(s) in {len(pairs)} annotation file(s).")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
