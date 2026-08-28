"""Measure the oracle voxel ceiling of a Gaussian primitive representation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.pengwin_eval import evaluate_f_level
from scripts.gaussian_primitives import load_primitive_set, splat_primitives_to_voxels


def _read_label(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path), (1.0, 1.0, 1.0)
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to read {path}") from exc
    image = sitk.ReadImage(str(path))
    spacing_xyz = image.GetSpacing()
    return sitk.GetArrayFromImage(image), (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))


def _write_label(path: str | Path, volume: np.ndarray, reference_path: str | Path | None = None) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, volume)
        return
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to write {path}") from exc
    image = sitk.GetImageFromArray(volume.astype(np.uint16))
    if reference_path is not None:
        ref = sitk.ReadImage(str(reference_path))
        image.CopyInformation(ref)
    sitk.WriteImage(image, str(path))


def compute_oracle_ceiling(
    primitive_path: str | Path,
    gt_path: str | Path,
    output_path: str | Path | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    primitive_set = load_primitive_set(primitive_path)
    if primitive_set.gt_match_ids is None:
        raise ValueError("ranked primitives must contain gt_match_ids for oracle ceiling")
    gt, image_spacing = _read_label(gt_path)
    spacing = image_spacing if spacing is None else spacing
    primitive_labels = primitive_set.gt_match_ids.astype(np.int32)
    foreground = gt > 0
    pred = splat_primitives_to_voxels(
        primitive_set,
        primitive_labels=primitive_labels,
        foreground_mask=foreground,
        target_range=(1, 65535),
    )
    if output_path is not None:
        _write_label(output_path, pred, reference_path=gt_path if str(gt_path).endswith(".mha") else None)
    metrics = evaluate_f_level(gt, pred, spacing=spacing)
    metrics.update(
        {
            "primitive_path": str(primitive_path),
            "gt_path": str(gt_path),
            "output_path": None if output_path is None else str(output_path),
            "num_primitives": int(len(primitive_set.centers_zyx)),
            "num_oracle_instances": int(len([x for x in np.unique(pred) if int(x) > 0])),
            "num_gt_instances": int(len([x for x in np.unique(gt) if int(x) > 0])),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--spacing-zyx", nargs=3, type=float, default=None)
    args = parser.parse_args()
    metrics = compute_oracle_ceiling(
        primitive_path=args.primitives,
        gt_path=args.gt,
        output_path=args.output,
        spacing=None if args.spacing_zyx is None else tuple(args.spacing_zyx),
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.summary is not None:
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
