"""Synthetic 3D label volumes with known geometric properties for testing."""
from __future__ import annotations

import numpy as np


def two_overlapping_cubes(
    shape: tuple[int, int, int] = (20, 20, 20),
    overlap_fraction: float = 0.5,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Two cubes whose intersection over union is approximately `overlap_fraction`.

    Returns (pred_label, gt_label, spacing). Both volumes have one fragment with
    ID=1. The known IoU is `overlap_fraction` (cubes are arranged so the
    intersection volume equals overlap_fraction * union_volume).
    """
    pred = np.zeros(shape, dtype=np.int32)
    gt = np.zeros(shape, dtype=np.int32)
    w = 10
    f = overlap_fraction
    k = int(round(w * (1 - f) / (1 + f)))
    gt[5:5 + w, 5:5 + w, 5:5 + w] = 1
    pred[5:5 + w, 5:5 + w, 5 + k:5 + k + w] = 1
    return pred, gt, spacing


def two_disjoint_cubes_distance_mm(
    distance_mm: float = 5.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Two single-voxel "cubes" separated by `distance_mm` along x."""
    pred = np.zeros((30, 30, 30), dtype=np.int32)
    gt = np.zeros((30, 30, 30), dtype=np.int32)
    voxel_dist = int(round(distance_mm / spacing[2]))
    gt[10, 10, 10] = 1
    pred[10, 10, 10 + voxel_dist] = 1
    return pred, gt, spacing


def n_disjoint_fragments(
    n: int = 3,
    anatomy_offset: int = 0,
    shape: tuple[int, int, int] = (30, 30, 30),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """`n` disjoint 3x3x3 cubes labelled sequentially from `anatomy_offset + 1`."""
    lbl = np.zeros(shape, dtype=np.int32)
    for i in range(n):
        x = 2 + i * 5
        lbl[2:5, 2:5, x:x + 3] = anatomy_offset + i + 1
    return lbl, spacing


def case_with_missing_fragment(
    n_gt: int = 3,
    n_pred: int = 2,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """GT has `n_gt` fragments; prediction keeps only the first `n_pred`."""
    gt, _ = n_disjoint_fragments(n_gt)
    pred = gt.copy()
    for i in range(n_pred, n_gt):
        pred[pred == i + 1] = 0
    return pred, gt, spacing


def perfect_prediction(
    n: int = 3,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Identical GT and pred — every metric should be at its best value."""
    lbl, _ = n_disjoint_fragments(n)
    return lbl.copy(), lbl, spacing
