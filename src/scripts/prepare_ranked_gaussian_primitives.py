"""Prepare ranked Gaussian primitives directly from CT and GT volumes."""
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

from scripts.fracture_atlas import build_fracture_atlas_from_labels
from scripts.gaussian_primitive_ranking import (
    attach_rank_result_to_primitives,
    rank_gaussian_primitives,
    topk_fragment_recall,
)
from scripts.gaussian_primitives import (
    attach_sacrum_anchor_features,
    attach_zero_sacrum_anchor_features,
    assign_primitive_anatomy_priors,
    compute_sacrum_anchor_frame,
    extract_gaussian_primitives,
    extract_gt_balanced_gaussian_primitives,
    save_primitive_set,
)
from scripts.rank_gaussian_primitives_case import _read_volume


def _read_spacing_zyx(path: str | Path) -> tuple[float, float, float]:
    path = Path(path)
    if path.suffix in {".npy", ".npz"}:
        return (1.0, 1.0, 1.0)
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError(f"SimpleITK is required to read spacing from {path}") from exc
    image = sitk.ReadImage(str(path))
    spacing_xyz = image.GetSpacing()
    return (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))


def _summary_from_ranked_case(
    gt_labels: np.ndarray,
    gt_match_ids: np.ndarray,
    rank_scores: np.ndarray,
    topk: int,
) -> dict[str, Any]:
    required_gt_ids = sorted(int(x) for x in np.unique(gt_labels) if int(x) > 0)
    positive_scores = rank_scores[gt_match_ids > 0]
    background_scores = rank_scores[gt_match_ids <= 0]
    return {
        "num_gt_fragments": int(len(required_gt_ids)),
        "num_primitives": int(len(rank_scores)),
        "topk": int(topk),
        "topk_fragment_recall": topk_fragment_recall(gt_match_ids, rank_scores, required_gt_ids, k=topk),
        "mean_rank_positive_gt": float(np.mean(positive_scores)) if len(positive_scores) else 0.0,
        "mean_rank_background": float(np.mean(background_scores)) if len(background_scores) else 0.0,
    }


def prepare_ranked_primitives_from_arrays(
    ct_hu: np.ndarray,
    gt_labels: np.ndarray,
    num_primitives: int = 128,
    radius: float = 4.0,
    topk: int = 32,
    balanced_gt: bool = False,
    min_primitives_per_fragment: int = 2,
    severity_aware: bool = False,
    spacing_zyx: tuple[float, float, float] | None = None,
    radius_units: str = "voxel",
    sacrum_anchor: bool = False,
) -> tuple[Any, dict[str, Any]]:
    ct = np.asarray(ct_hu, dtype=np.float32)
    gt = np.asarray(gt_labels)
    if ct.shape != gt.shape:
        raise ValueError(f"CT shape {ct.shape} does not match GT shape {gt.shape}")
    foreground_mask = gt > 0
    fragment_importance = None
    fragment_severity = None
    if severity_aware:
        atlas = build_fracture_atlas_from_labels({"case": gt})
        fragment_importance = {fragment.fragment_id: fragment.importance_weight for fragment in atlas.fragments}
        fragment_severity = {fragment.fragment_id: fragment.severity_label for fragment in atlas.fragments}
    if balanced_gt:
        primitive_set = extract_gt_balanced_gaussian_primitives(
            ct,
            gt,
            num_primitives=num_primitives,
            min_primitives_per_fragment=min_primitives_per_fragment,
            radius=radius,
            fragment_importance=fragment_importance,
            fragment_severity=fragment_severity,
            spacing_zyx=spacing_zyx,
            radius_units=radius_units,
        )
    else:
        primitive_set = extract_gaussian_primitives(
            ct,
            foreground_mask,
            num_primitives=num_primitives,
            radius=radius,
            spacing_zyx=spacing_zyx,
            radius_units=radius_units,
        )
    sacrum_anchor_status = "disabled"
    if sacrum_anchor:
        try:
            frame = compute_sacrum_anchor_frame(gt, spacing_zyx=spacing_zyx or (1.0, 1.0, 1.0))
        except ValueError:
            primitive_set = attach_zero_sacrum_anchor_features(primitive_set)
            sacrum_anchor_status = "missing_zero"
        else:
            primitive_set = attach_sacrum_anchor_features(primitive_set, frame)
            sacrum_anchor_status = "attached"
    primitive_set = assign_primitive_anatomy_priors(primitive_set, volume_shape=gt.shape)
    rank_result = rank_gaussian_primitives(primitive_set, gt, ct)
    ranked = attach_rank_result_to_primitives(primitive_set, rank_result)
    summary = _summary_from_ranked_case(gt, rank_result.gt_match_ids, rank_result.rank_scores, topk=topk)
    summary["severity_aware"] = bool(severity_aware)
    summary["radius_units"] = radius_units
    summary["spacing_zyx"] = [float(x) for x in (spacing_zyx or (1.0, 1.0, 1.0))]
    summary["sacrum_anchor"] = bool(sacrum_anchor)
    summary["sacrum_anchor_status"] = sacrum_anchor_status
    return ranked, summary


def prepare_ranked_primitives_from_paths(
    ct_path: str | Path,
    gt_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    num_primitives: int = 128,
    radius: float = 4.0,
    topk: int = 32,
    balanced_gt: bool = False,
    min_primitives_per_fragment: int = 2,
    severity_aware: bool = False,
    radius_units: str = "voxel",
    sacrum_anchor: bool = False,
) -> dict[str, Any]:
    ct_hu = _read_volume(ct_path).astype(np.float32)
    gt_labels = _read_volume(gt_path).astype(np.int32)
    spacing_zyx = _read_spacing_zyx(ct_path)
    ranked, summary = prepare_ranked_primitives_from_arrays(
        ct_hu=ct_hu,
        gt_labels=gt_labels,
        num_primitives=num_primitives,
        radius=radius,
        topk=topk,
        balanced_gt=balanced_gt,
        min_primitives_per_fragment=min_primitives_per_fragment,
        severity_aware=severity_aware,
        spacing_zyx=spacing_zyx,
        radius_units=radius_units,
        sacrum_anchor=sacrum_anchor,
    )
    save_primitive_set(output_path, ranked)
    if summary_path is not None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, help="CT volume path: .npy, .npz, .nii.gz, .mha")
    parser.add_argument("--gt", required=True, help="GT label volume path: .npy, .npz, .nii.gz, .mha")
    parser.add_argument("--output", required=True, help="Output ranked primitive .npz")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    parser.add_argument("--num-primitives", type=int, default=128)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--balanced-gt", action="store_true", help="Guarantee a minimum primitive quota per GT fragment")
    parser.add_argument("--min-primitives-per-fragment", type=int, default=2)
    parser.add_argument("--severity-aware", action="store_true", help="Attach severity labels and importance weights")
    parser.add_argument("--radius-units", default="voxel", choices=("voxel", "mm"))
    parser.add_argument("--sacrum-anchor", action="store_true", help="Attach sacrum-fixed relative coordinate features")
    args = parser.parse_args()
    summary = prepare_ranked_primitives_from_paths(
        ct_path=args.ct,
        gt_path=args.gt,
        output_path=args.output,
        summary_path=args.summary,
        num_primitives=args.num_primitives,
        radius=args.radius,
        topk=args.topk,
        balanced_gt=args.balanced_gt,
        min_primitives_per_fragment=args.min_primitives_per_fragment,
        severity_aware=args.severity_aware,
        radius_units=args.radius_units,
        sacrum_anchor=args.sacrum_anchor,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
