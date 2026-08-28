"""GT-free CT-derived anchor priors for PENGWIN CT volumes.

The original SAFT prototype seeded the coordinate frame on the *ground-truth*
sacrum centroid (`fragment_label` ids 1..50). That label does not exist at
inference, so training fed the model 5/6 input channels it could never see at
test time -- a label-leakage + covariate-shift defect that collapsed test
anatomy Dice to ~0.

This version derives the anchor centroid from the CT bone intensity instead, so
the identical prior is computable at training and at inference from the image
alone. The same `build_anchor_channels(image, ...)` must be used in both phases.
"""
from __future__ import annotations

import numpy as np


def estimate_bone_centroid_zyx(image_zyx: np.ndarray, bone_threshold: float = 0.1) -> np.ndarray:
    """Centroid (Z/Y/X voxels) of dense-bone voxels in a HU-normalized CT.

    `image_zyx` is expected in the `normalize_hu` range (~[-1, 1]); cortical bone
    sits above ~0.1 (~650 HU), well clear of soft tissue and of the 0.0 padding
    value used when cropping patches. Falls back to the volume center when no
    bone voxel is present so the prior is always GT-free and well-defined.
    """
    image = np.asarray(image_zyx, dtype=np.float32)
    bone = image > float(bone_threshold)
    if bone.any():
        return np.argwhere(bone).mean(axis=0).astype(np.float32)
    return ((np.array(image.shape, dtype=np.float32) - 1.0) * 0.5).astype(np.float32)


def build_anchor_channels(
    image_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    clip_mm: float = 200.0,
    bone_threshold: float = 0.1,
) -> np.ndarray:
    """Build 5 GT-free channels: relative z/y/x, distance-to-anchor, side prior.

    The anchor is the CT bone centroid (a GT-free proxy for the pelvic center).
    Output shape is (5, Z, Y, X), aligned to `image_zyx`.
    """
    shape = tuple(int(x) for x in image_zyx.shape)
    centroid = estimate_bone_centroid_zyx(image_zyx, bone_threshold)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    grid = np.indices(shape, dtype=np.float32)
    rel = (grid - centroid[:, None, None, None]) * spacing[:, None, None, None]
    rel_norm = np.clip(rel / float(clip_mm), -1.0, 1.0)
    dist = np.linalg.norm(rel, axis=0, keepdims=True)
    dist_norm = np.clip(dist / float(clip_mm), 0.0, 1.0)
    # X-axis sign is a lightweight side prior in the anchor-centered frame.
    side = rel_norm[2:3]
    return np.concatenate([rel_norm, dist_norm, side], axis=0).astype(np.float32, copy=False)
