"""Extra instance/topology metrics for SAFT-GD hardcase evaluation."""
from __future__ import annotations

import numpy as np

from evaluation.matching import compute_pairwise_iou


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else 0.0


def evaluate_instance_quality(
    gt_lbl: np.ndarray,
    pred_lbl: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute instance F1/recall/precision and merge/split topology errors."""
    gt_ids = sorted(int(x) for x in set(np.unique(gt_lbl)) - {0})
    pred_ids = sorted(int(x) for x in set(np.unique(pred_lbl)) - {0})
    iou = compute_pairwise_iou(gt_lbl, pred_lbl, gt_ids, pred_ids)
    if not gt_ids and not pred_ids:
        return {
            "instance_f1": 1.0,
            "instance_recall": 1.0,
            "instance_precision": 1.0,
            "merge_errors": 0.0,
            "split_errors": 0.0,
            "topology_consistency": 1.0,
        }
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    candidates = sorted(
        ((float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])),
        reverse=True,
    )
    for value, i, j in candidates:
        if value < iou_threshold:
            break
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
    tp = len(matched_gt)
    fn = len(gt_ids) - tp
    fp = len(pred_ids) - tp
    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    overlap = iou > 0
    merge_errors = 0
    for j in range(len(pred_ids)):
        if int(overlap[:, j].sum()) > 1:
            merge_errors += int(overlap[:, j].sum()) - 1
    split_errors = 0
    for i in range(len(gt_ids)):
        if int(overlap[i, :].sum()) > 1:
            split_errors += int(overlap[i, :].sum()) - 1
    topology_penalty = merge_errors + split_errors
    topology_consistency = _safe_div(max(len(gt_ids), len(pred_ids)) - topology_penalty, max(len(gt_ids), len(pred_ids)))
    return {
        "instance_f1": float(f1),
        "instance_recall": float(recall),
        "instance_precision": float(precision),
        "merge_errors": float(merge_errors),
        "split_errors": float(split_errors),
        "topology_consistency": float(max(0.0, min(1.0, topology_consistency))),
    }
