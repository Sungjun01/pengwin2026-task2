"""Topology-preserving label surface refinement.

This post-processing step is intentionally conservative: it may trim low-HU
exterior fringes or recover adjacent high-HU shell voxels, but it must not
merge, split, or rename instances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class SurfaceRefineConfig:
    trim_hu_threshold: float = 50.0
    recover_hu_threshold: float = 300.0
    max_iters: int = 1
    min_keep_voxels: int = 8


def _ids(labels: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(labels) if int(x) != 0)


def refine_instance_surfaces(
    labels: np.ndarray,
    ct_hu: np.ndarray,
    cfg: SurfaceRefineConfig | None = None,
) -> np.ndarray:
    """Refine boundaries while preserving the nonzero instance ID set."""
    cfg = cfg or SurfaceRefineConfig()
    if labels.shape != ct_hu.shape:
        raise ValueError(f"ct_hu shape {ct_hu.shape} does not match labels shape {labels.shape}")

    out = labels.copy()
    original_ids = _ids(labels)
    for iid in original_ids:
        mask = out == iid
        if not mask.any():
            continue

        for _ in range(max(1, int(cfg.max_iters))):
            mask = out == iid
            if int(mask.sum()) <= cfg.min_keep_voxels:
                break

            interior = binary_erosion(mask, structure=STRUCT26, iterations=1)
            surface = mask & ~interior
            trim = surface & (ct_hu < cfg.trim_hu_threshold)
            if trim.any() and int(mask.sum() - trim.sum()) >= cfg.min_keep_voxels:
                out[trim] = 0

            mask = out == iid
            candidate = binary_dilation(mask, structure=STRUCT26, iterations=1) & (out == 0)
            recover = candidate & (ct_hu >= cfg.recover_hu_threshold)
            if recover.any():
                out[recover] = iid

    # If a very small instance was accidentally trimmed away, restore it.
    for iid in original_ids:
        if not np.any(out == iid):
            out[labels == iid] = iid
    return out.astype(labels.dtype, copy=False)
