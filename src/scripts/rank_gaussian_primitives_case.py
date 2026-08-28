"""Attach GT-supervised Gaussian rank scores to a primitive artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.gaussian_primitive_ranking import (
    attach_rank_result_to_primitives,
    rank_gaussian_primitives,
    topk_fragment_recall,
)
from scripts.gaussian_primitives import load_primitive_set, save_primitive_set


def _read_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        if "volume" in data:
            return data["volume"]
        raise ValueError(f"npz volume must contain arr_0 or volume: {path}")
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError(f"SimpleITK is required to read non-numpy volume: {path}") from exc
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _ranking_summary(
    gt_labels: np.ndarray,
    gt_match_ids: np.ndarray,
    rank_scores: np.ndarray,
    topk: int,
) -> dict[str, Any]:
    required_gt_ids = sorted(int(x) for x in np.unique(gt_labels) if int(x) > 0)
    positive_scores = rank_scores[gt_match_ids > 0]
    background_scores = rank_scores[gt_match_ids <= 0]
    return {
        "num_primitives": int(len(rank_scores)),
        "num_gt_fragments": int(len(required_gt_ids)),
        "topk": int(topk),
        "topk_fragment_recall": topk_fragment_recall(gt_match_ids, rank_scores, required_gt_ids, topk),
        "mean_rank_positive_gt": float(np.mean(positive_scores)) if len(positive_scores) else 0.0,
        "mean_rank_background": float(np.mean(background_scores)) if len(background_scores) else 0.0,
        "ranked_gt_match_ids": [int(x) for x in gt_match_ids[np.argsort(rank_scores)[::-1]].tolist()],
    }


def rank_case_from_paths(
    primitive_path: str | Path,
    ct_path: str | Path,
    gt_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    topk: int = 32,
) -> dict[str, Any]:
    primitive_set = load_primitive_set(primitive_path)
    ct_hu = _read_volume(ct_path).astype(np.float32)
    gt_labels = _read_volume(gt_path).astype(np.int32)
    if ct_hu.shape != gt_labels.shape:
        raise ValueError(f"CT shape {ct_hu.shape} does not match GT shape {gt_labels.shape}")
    result = rank_gaussian_primitives(primitive_set, gt_labels, ct_hu)
    ranked = attach_rank_result_to_primitives(primitive_set, result)
    save_primitive_set(output_path, ranked)
    summary = _ranking_summary(gt_labels, result.gt_match_ids, result.rank_scores, topk=topk)
    if summary_path is not None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", required=True, help="Input primitive .npz")
    parser.add_argument("--ct", required=True, help="CT volume path: .npy, .npz, .nii.gz, .mha")
    parser.add_argument("--gt", required=True, help="GT label volume path: .npy, .npz, .nii.gz, .mha")
    parser.add_argument("--output", required=True, help="Output ranked primitive .npz")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()
    summary = rank_case_from_paths(
        primitive_path=args.primitives,
        ct_path=args.ct,
        gt_path=args.gt,
        output_path=args.output,
        summary_path=args.summary,
        topk=args.topk,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
