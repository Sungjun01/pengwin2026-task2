"""F-CIRL: Aseg loss adapted from CIRL (Chen et al., CVPR 2024, crack segmentation).

Aseg loss (paper Eq. 3/7, Wasserstein derivation):
    L = mean[ (y_p - y_gt)^2 + sigma^2 ] / [ lambda + sigma^2 ]
where sigma^2 = exp(log_var) from a per-voxel std head (predict log_var for
numerical stability). Intuition: being wrong is OK *if* the model admits high
sigma there; being wrong while confident (low sigma) is penalized.

Our fracture adaptation (F-CIRL):
- anatomy-constrained: variance is only active on channels of the SAME anatomy
  as the voxel — cross-anatomy uncertainty is penalized (preserve anatomy IoU).
- spatially restricted to a marginal mask (the eval Local-Dice 20mm region).

This module is the representation-agnostic core (loss + channel mask). The
std head (CentroidConditionedStdHead) and nnUNet trainer integration are added
only after the Task 3 gate confirms the border-core representation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# 7-class border-core: class -> anatomy id (0 bg, 1 sacrum, 2 LH, 3 RH)
CLASS_TO_ANATOMY = (0, 1, 1, 2, 2, 3, 3)

# pelvic anatomy id ranges (PENGWIN instance encoding)
PELVIC_ANATOMY_RANGES = ((1, 50), (51, 100), (101, 150))


def compute_marginal_mask_v2(instance: np.ndarray,
                             spacing,
                             anatomy_ranges=PELVIC_ANATOMY_RANGES,
                             radius_mm: float = 20.0) -> np.ndarray:
    """Voxels within `radius_mm` of a same-anatomy inter-fragment seam.

    This is the spatial region where the Aseg loss applies — the eval Local-Dice
    20 mm band around fragment contact surfaces (spec §15). Computed per-case from
    the GT instance volume (PENGWIN IDs 1-200).

    instance : (D,H,W) int  — PENGWIN instance ids
    spacing  : (sz, sy, sx) mm per voxel along (D,H,W) axes
    returns  : (D,H,W) bool — marginal mask
    """
    seam = np.zeros(instance.shape, dtype=bool)
    for lo, hi in anatomy_ranges:
        a = np.where((instance >= lo) & (instance <= hi), instance, 0)
        for axis in range(3):
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[axis] = slice(0, -1)
            sl_hi[axis] = slice(1, None)
            lo_a = a[tuple(sl_lo)]
            hi_a = a[tuple(sl_hi)]
            # contact between two DIFFERENT fragments of this anatomy (no wraparound)
            contact = (lo_a > 0) & (hi_a > 0) & (lo_a != hi_a)
            seam[tuple(sl_lo)] |= contact
            seam[tuple(sl_hi)] |= contact
    if not seam.any():
        return np.zeros(instance.shape, dtype=bool)
    dist_mm = distance_transform_edt(~seam, sampling=spacing)
    return dist_mm <= radius_mm


def build_same_anatomy_channel_mask(gt_anatomy_per_voxel: torch.Tensor,
                                    num_classes: int = 7) -> torch.Tensor:
    """(B,D,H,W) anatomy ids {0,1,2,3} -> (B,num_classes,D,H,W) bool.

    True at (b,c,...) iff class c belongs to the same anatomy as voxel (b,...).
    """
    c2a = torch.tensor(CLASS_TO_ANATOMY[:num_classes], device=gt_anatomy_per_voxel.device)
    voxel_anat = gt_anatomy_per_voxel.unsqueeze(1)            # (B,1,D,H,W)
    chan_anat = c2a.view(1, num_classes, 1, 1, 1)             # (1,C,1,1,1)
    return voxel_anat == chan_anat


def marginal_mask_from_label_torch(label: torch.Tensor,
                                   border_classes=(2, 4, 6),
                                   iters: int = 25) -> torch.Tensor:
    """GPU marginal mask for the trainer: dilate the GT border classes by `iters`
    voxels (each ≈ 1 voxel via 3³ max-pool). The border-core border class IS the
    inter-fragment seam, so this is the on-the-fly equivalent of compute_marginal_mask_v2
    without needing the instance GT. `iters` ≈ radius_mm / mean_spacing_mm.

    label : (B,1,D,H,W) or (B,D,H,W) int class map. returns (B,D,H,W) bool.
    """
    if label.dim() == 5:
        label = label[:, 0]
    border = torch.zeros_like(label, dtype=torch.bool)
    for c in border_classes:
        border |= (label == c)
    m = border.float().unsqueeze(1)
    for _ in range(iters):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m.squeeze(1) > 0


class CentroidConditionedStdHead(nn.Module):
    """Per-voxel log-variance head for the Aseg loss (spec §15).

    Input = [decoder last feature ; centroid-offset channels]; a 1x1x1 conv maps it
    to `num_classes` log_var values. The offset channels are an uncertainty *prior*:
    near the LH/RH midline the sacrum-centroid offset_x ≈ 0 (ambiguous side), so the
    head can raise sigma there. Zero-initialised → log_var starts at 0 (sigma²=1),
    which keeps early training stable before the head learns where to be uncertain.

    Inference never calls this head, so it adds zero test-time cost.
    """

    def __init__(self, in_channels: int, num_classes: int = 7, offset_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels + offset_channels, num_classes, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, feature: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """feature (B, Cf, D, H, W), offset (B, 3, D, H, W) -> log_var (B, C, D, H, W)."""
        return self.conv(torch.cat([feature, offset], dim=1))


def aseg_loss_anatomy_constrained(seg_logits: torch.Tensor,
                                  log_var: torch.Tensor,
                                  gt_onehot: torch.Tensor,
                                  marginal_mask: torch.Tensor,
                                  gt_anatomy_per_voxel: torch.Tensor,
                                  lambda_: float = 0.5) -> torch.Tensor:
    """Anatomy-constrained, spatially-masked Aseg loss.

    seg_logits, log_var, gt_onehot : (B, C, D, H, W)
    marginal_mask                  : (B, D, H, W) bool  — where Aseg applies
    gt_anatomy_per_voxel           : (B, D, H, W) int    — anatomy id per voxel
    """
    num_classes = seg_logits.shape[1]
    seg_soft = F.softmax(seg_logits, dim=1)
    sigma2 = torch.exp(log_var)
    sq_err = (seg_soft - gt_onehot) ** 2

    chan_mask = build_same_anatomy_channel_mask(gt_anatomy_per_voxel, num_classes)
    sigma2 = sigma2 * chan_mask.to(sigma2.dtype)             # cross-anatomy sigma -> 0

    pixel_loss = (sq_err + sigma2) / (lambda_ + sigma2)      # (B,C,D,H,W)

    m = marginal_mask.unsqueeze(1).expand_as(pixel_loss).to(pixel_loss.dtype)
    return (pixel_loss * m).sum() / (m.sum() + 1e-6)
