"""Fragment matching: pair predicted instances to ground-truth instances."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_pairwise_iou(
    gt_lbl: np.ndarray,
    pred_lbl: np.ndarray,
    gt_ids: list[int],
    pred_ids: list[int],
) -> np.ndarray:
    """Compute IoU matrix of shape [|gt_ids|, |pred_ids|].

    Each entry [i, j] is IoU(gt_lbl == gt_ids[i], pred_lbl == pred_ids[j]).
    """
    if not gt_ids or not pred_ids:
        return np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)

    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_masks = [(gt_lbl == g) for g in gt_ids]
    pred_masks = [(pred_lbl == p) for p in pred_ids]
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            inter = int(np.logical_and(gm, pm).sum())
            union = int(np.logical_or(gm, pm).sum())
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def _hungarian_from_iou(
    iou: np.ndarray, gt_ids: list[int], pred_ids: list[int]
) -> dict[int, int | None]:
    """Optimal one-to-one assignment maximizing total IoU."""
    if iou.size == 0:
        return {g: None for g in gt_ids}
    gt_idx, pred_idx = linear_sum_assignment(-iou)
    matching: dict[int, int | None] = {g: None for g in gt_ids}
    for i, j in zip(gt_idx, pred_idx):
        if iou[i, j] > 0:
            matching[gt_ids[i]] = pred_ids[j]
    return matching


def _greedy_from_iou(
    iou: np.ndarray, gt_ids: list[int], pred_ids: list[int]
) -> dict[int, int | None]:
    """Greedy one-to-one: repeatedly pick (gt, pred) with the largest IoU."""
    matching: dict[int, int | None] = {g: None for g in gt_ids}
    if iou.size == 0:
        return matching
    flat = sorted(
        ((iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for v, i, j in flat:
        if v <= 0:
            break
        if i in used_gt or j in used_pred:
            continue
        matching[gt_ids[i]] = pred_ids[j]
        used_gt.add(i)
        used_pred.add(j)
    return matching


def match_fragments(
    gt_lbl: np.ndarray,
    pred_lbl: np.ndarray,
    method: Literal["hungarian", "greedy"] = "hungarian",
) -> dict[int, int | None]:
    """Return {gt_fragment_id: pred_fragment_id or None if unmatched}.

    PENGWIN paper §3.3.1 specifies "highest IoU with one-to-one correspondence".
    'hungarian' (default) gives the optimal one-to-one assignment.
    'greedy' is provided for ablation but may be suboptimal.
    """
    gt_ids = sorted(int(x) for x in set(np.unique(gt_lbl)) - {0})
    pred_ids = sorted(int(x) for x in set(np.unique(pred_lbl)) - {0})
    if not gt_ids:
        return {}
    iou = compute_pairwise_iou(gt_lbl, pred_lbl, gt_ids, pred_ids)
    if method == "hungarian":
        return _hungarian_from_iou(iou, gt_ids, pred_ids)
    elif method == "greedy":
        return _greedy_from_iou(iou, gt_ids, pred_ids)
    else:
        raise ValueError(f"Unknown matching method: {method!r}")
