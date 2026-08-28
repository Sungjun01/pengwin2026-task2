"""Primitive segmentation metrics on binary masks."""
from __future__ import annotations

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union for two binary 3-D masks.

    Returns 0.0 when both masks are empty (consistent with PENGWIN convention
    where missing predictions get zero overlap).
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return intersection / union if union > 0 else 0.0


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice (Sørensen-Dice) coefficient = 2|A∩B| / (|A|+|B|).

    Returns 0.0 when both masks are empty (missing-fragment penalty 흐름과 정합).
    PENGWIN 2026 의 LDSC (Local Dice Similarity Coefficient) per-fragment 계산
    의 base building block.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    size_sum = int(pred.sum()) + int(gt.sum())
    return (2.0 * intersection / size_sum) if size_sum > 0 else 0.0


def recognition_quality(
    tp: int, fp: int, fn: int,
) -> float:
    """Recognition Quality (panoptic 표준) = TP / (TP + 0.5*FP + 0.5*FN).

    매칭 흐름 (PENGWIN 2026 §evaluation).
        TP : matched (gt, pred) pair with IoU >= threshold
        FN : unmatched GT 또는 matched-with-IoU<threshold 의 GT side
        FP : unmatched pred 또는 matched-with-IoU<threshold 의 pred side

    panoptic 정의에서 matched-but-low-IoU 는 FP / FN 양쪽에 카운트.
    """
    denom = tp + 0.5 * fp + 0.5 * fn
    return (tp / denom) if denom > 0 else 0.0


def bounding_sphere_diameter_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Diameter of the volume-equivalent sphere of a binary mask, in mm.

    Used as the PENGWIN missing-fragment penalty for HD95 (and radius for ASSD).
    Formula: d = 2 * (3V / 4π)^(1/3) where V = voxel_count * prod(spacing).
    Returns 0.0 for empty masks.
    """
    voxel_count = int(mask.astype(bool).sum())
    if voxel_count == 0:
        return 0.0
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    volume_mm3 = voxel_count * voxel_volume
    radius = (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return 2.0 * radius


def _to_monai_tensors(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert two 3-D binary masks to [B=1, C=1, D, H, W] float tensors."""
    pred = torch.from_numpy(pred_mask.astype(np.float32))[None, None]
    gt = torch.from_numpy(gt_mask.astype(np.float32))[None, None]
    return pred, gt


def hd95_mm(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """95th percentile Hausdorff distance in mm."""
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    pred, gt = _to_monai_tensors(pred_mask, gt_mask)
    d = compute_hausdorff_distance(
        pred, gt,
        include_background=False,
        percentile=95,
        spacing=spacing,
    )
    return float(d.item())


def assd_mm(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Average Symmetric Surface Distance in mm."""
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    pred, gt = _to_monai_tensors(pred_mask, gt_mask)
    d = compute_average_surface_distance(
        pred, gt,
        include_background=False,
        symmetric=True,
        spacing=spacing,
    )
    return float(d.item())
