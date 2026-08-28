"""PENGWIN per-case evaluation: F-level + A-level metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from evaluation.io_utils import load_label_volume, pair_predictions_with_gt
from evaluation.label_schema import BONE_RANGES, remap_to_anatomy
from evaluation.matching import match_fragments
from evaluation.metrics import (
    iou, hd95_mm, assd_mm, bounding_sphere_diameter_mm, dice, recognition_quality,
)


SCHEMA_VERSION = "1.1"
RQ_IOU_THRESHOLD = 0.5  # panoptic 표준. PENGWIN 2026 organizer 가 다르게 명시하면 조정.

# 공식 평가: "Before final metric computation, small connected components are
# removed using a case-specific voxel threshold corresponding to approximately
# 1 cm³ (=1000 mm³)." 우리 로컬 채점도 이를 맞춰야 slivers 가 FP 로 과대계상되지
# 않는다. 기본 None = 끔(기존 스크립트 결과 보존), 1000 = 공식 정합.
OFFICIAL_MIN_CC_MM3 = 1000.0


def remove_small_components(
    lbl: np.ndarray, spacing_zyx: tuple[float, float, float], min_mm3: float
) -> np.ndarray:
    """각 instance 라벨의 connected component 중 min_mm3 미만은 0(배경)으로 제거.
    공식 평가의 small-CC cleanup 재현 (pred·gt 동일 적용해 일관성 유지)."""
    from scipy.ndimage import label as _cc, generate_binary_structure
    if min_mm3 <= 0:
        return lbl
    voxel_mm3 = float(np.prod(spacing_zyx))
    min_vox = min_mm3 / max(voxel_mm3, 1e-9)
    st = generate_binary_structure(3, 3)
    out = lbl.copy()
    for fid in (int(x) for x in set(np.unique(lbl)) - {0}):
        m = lbl == fid
        cc, n = _cc(m, structure=st)
        if n <= 1:
            if int(m.sum()) < min_vox:
                out[m] = 0
            continue
        for c in range(1, n + 1):
            piece = cc == c
            if int(piece.sum()) < min_vox:
                out[piece] = 0
    return out


def evaluate_f_level(
    gt_lbl: np.ndarray,
    pred_lbl: np.ndarray,
    spacing: tuple[float, float, float],
    matching_method: Literal["hungarian", "greedy"] = "hungarian",
    rq_iou_threshold: float = RQ_IOU_THRESHOLD,
) -> dict[str, float]:
    """F-level: per-fragment matched eval. Missing fragments penalized via
    bounding sphere (HD95 = diameter, ASSD = radius). LDSC = mean Dice over
    GT fragments (unmatched 는 0 penalty). RQ = panoptic Recognition Quality.
    Returns NaN-filled dict when GT has no fragments."""
    matching = match_fragments(gt_lbl, pred_lbl, method=matching_method)
    if not matching:
        return {
            "IoU_F": float("nan"),
            "HD95_F_mm": float("nan"),
            "ASSD_F_mm": float("nan"),
            "LDSC_F": float("nan"),
            "RQ_F": float("nan"),
        }

    iou_vals: list[float] = []
    hd95_vals: list[float] = []
    assd_vals: list[float] = []
    dice_vals: list[float] = []

    tp = 0
    fn_count = 0
    fp_low_iou = 0  # matched-but-low-IoU pred side
    matched_pred_ids: set[int] = set()

    for gt_id, pred_id in matching.items():
        gt_mask = (gt_lbl == gt_id)
        if pred_id is None:
            d = bounding_sphere_diameter_mm(gt_mask, spacing)
            iou_vals.append(0.0)
            hd95_vals.append(d)
            assd_vals.append(d / 2.0)
            dice_vals.append(0.0)
            fn_count += 1
        else:
            pred_mask = (pred_lbl == pred_id)
            iou_v = iou(pred_mask, gt_mask)
            iou_vals.append(iou_v)
            hd95_vals.append(hd95_mm(pred_mask, gt_mask, spacing))
            assd_vals.append(assd_mm(pred_mask, gt_mask, spacing))
            dice_vals.append(dice(pred_mask, gt_mask))
            matched_pred_ids.add(int(pred_id))
            if iou_v >= rq_iou_threshold:
                tp += 1
            else:
                fn_count += 1
                fp_low_iou += 1

    all_pred_ids = {int(x) for x in set(np.unique(pred_lbl)) - {0}}
    unmatched_pred = all_pred_ids - matched_pred_ids
    fp_count = len(unmatched_pred) + fp_low_iou

    return {
        "IoU_F": float(np.mean(iou_vals)),
        "HD95_F_mm": float(np.mean(hd95_vals)),
        "ASSD_F_mm": float(np.mean(assd_vals)),
        "LDSC_F": float(np.mean(dice_vals)),
        "RQ_F": float(recognition_quality(tp, fp_count, fn_count)),
    }


def evaluate_a_level(
    gt_lbl: np.ndarray,
    pred_lbl: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """A-level: merge fragments by anatomy class, compute per-class metrics,
    average over classes present in GT.

    Returns NaN-filled dict when no anatomy class is present in GT.
    """
    gt_anat = remap_to_anatomy(gt_lbl)
    pred_anat = remap_to_anatomy(pred_lbl)
    iou_vals: list[float] = []
    hd95_vals: list[float] = []
    assd_vals: list[float] = []
    for cls in sorted(BONE_RANGES.keys()):
        gt_mask = (gt_anat == cls)
        if not gt_mask.any():
            continue
        pred_mask = (pred_anat == cls)
        iou_vals.append(iou(pred_mask, gt_mask))
        hd95_vals.append(hd95_mm(pred_mask, gt_mask, spacing))
        assd_vals.append(assd_mm(pred_mask, gt_mask, spacing))
    if not iou_vals:
        return {
            "IoU_A": float("nan"),
            "HD95_A_mm": float("nan"),
            "ASSD_A_mm": float("nan"),
        }
    return {
        "IoU_A": float(np.mean(iou_vals)),
        "HD95_A_mm": float(np.mean(hd95_vals)),
        "ASSD_A_mm": float(np.mean(assd_vals)),
    }


def evaluate_case(
    pred_path: Path | str,
    gt_path: Path | str,
    matching_method: Literal["hungarian", "greedy"] = "hungarian",
    min_cc_mm3: float | None = None,
) -> dict[str, float]:
    """Six PENGWIN metrics for one (pred, gt) file pair.

    min_cc_mm3: 공식 평가의 small-CC cleanup (≈1000mm³). None=끔(기존 동작 보존)."""
    pred_lbl, _ = load_label_volume(pred_path)
    gt_lbl, gt_spacing = load_label_volume(gt_path)
    if min_cc_mm3:
        pred_lbl = remove_small_components(pred_lbl, gt_spacing, min_cc_mm3)
        gt_lbl = remove_small_components(gt_lbl, gt_spacing, min_cc_mm3)
    f = evaluate_f_level(gt_lbl, pred_lbl, gt_spacing, matching_method)
    a = evaluate_a_level(gt_lbl, pred_lbl, gt_spacing)
    gt_frag_ids = sorted(int(x) for x in set(np.unique(gt_lbl)) - {0})
    pred_frag_ids = sorted(int(x) for x in set(np.unique(pred_lbl)) - {0})
    return {
        **f, **a,
        "n_gt_fragments": len(gt_frag_ids),
        "n_pred_fragments": len(pred_frag_ids),
    }


def _aggregate(per_case: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean / median / std / min / max over cases, per metric."""
    if not per_case:
        return {}
    metric_keys = [k for k in next(iter(per_case.values())).keys()
                   if k not in ("n_gt_fragments", "n_pred_fragments")]
    agg: dict[str, float] = {}
    for k in metric_keys:
        vals = np.array(
            [c[k] for c in per_case.values() if not np.isnan(c[k])],
            dtype=np.float64,
        )
        if vals.size == 0:
            for stat in ("mean", "median", "std", "min", "max"):
                agg[f"{k}_{stat}"] = float("nan")
            continue
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_median"] = float(np.median(vals))
        agg[f"{k}_std"] = float(vals.std(ddof=0))
        agg[f"{k}_min"] = float(vals.min())
        agg[f"{k}_max"] = float(vals.max())
    return agg


def evaluate_submission(
    pred_dir: Path | str,
    gt_dir: Path | str,
    output_path: Path | str | None = None,
    matching_method: Literal["hungarian", "greedy"] = "hungarian",
    min_cc_mm3: float | None = None,
) -> dict:
    """Run evaluate_case across every (pred, gt) pair under the dirs and
    aggregate. Writes JSON output if output_path is given.

    min_cc_mm3: 공식 평가의 small-CC cleanup (≈1000mm³). None=끔(기존 동작 보존)."""
    pairs = pair_predictions_with_gt(pred_dir, gt_dir)
    per_case: dict[str, dict[str, float]] = {}
    for pred_path, gt_path in pairs:
        name = pred_path.name
        for suffix in (".nii.gz", ".mha", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        per_case[name] = evaluate_case(pred_path, gt_path, matching_method, min_cc_mm3)
    result = {
        "schema_version": SCHEMA_VERSION,
        "matching_method": matching_method,
        "min_cc_mm3": min_cc_mm3,
        "per_case": per_case,
        "aggregate": _aggregate(per_case),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="PENGWIN per-submission evaluation (F-level + A-level).",
    )
    parser.add_argument("-p", "--pred_dir", required=True, type=Path)
    parser.add_argument("-g", "--gt_dir", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--matching", choices=["hungarian", "greedy"],
                        default="hungarian")
    args = parser.parse_args()
    result = evaluate_submission(
        args.pred_dir, args.gt_dir, args.output, matching_method=args.matching,
    )
    print(f"Evaluated {len(result['per_case'])} cases.")
    agg = result["aggregate"]
    print(f"IoU_F mean:   {agg.get('IoU_F_mean', float('nan')):.4f}")
    print(f"HD95_F mean:  {agg.get('HD95_F_mm_mean', float('nan')):.4f} mm")
    print(f"ASSD_F mean:  {agg.get('ASSD_F_mm_mean', float('nan')):.4f} mm")
    print(f"LDSC_F mean:  {agg.get('LDSC_F_mean', float('nan')):.4f}")
    print(f"RQ_F mean:    {agg.get('RQ_F_mean', float('nan')):.4f}")
    print(f"IoU_A mean:   {agg.get('IoU_A_mean', float('nan')):.4f}")
    print(f"HD95_A mean:  {agg.get('HD95_A_mm_mean', float('nan')):.4f} mm")
    print(f"ASSD_A mean:  {agg.get('ASSD_A_mm_mean', float('nan')):.4f} mm")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    _main()
