"""Topology-aware border losses for the separation root-fix (design 2026-06-10).

Diagnosis (OOF170 gate + report_102): 81% of cases are separation failures, and
on the seams v18 merges through, the model predicts border on only ~54% of voxels
and *core* on ~42% — a discontinuous border band that the 26-conn decode bridges.
Per-voxel weighted-CE cannot see this: a few missed border voxels that form a
bridge are catastrophic for instances but cheap for CE.

Two complementary train-only terms (inference graph unchanged — these touch only
the loss, never the seg head):

  1. clDice border-continuity (`cldice_border_loss`): forces the predicted border
     foreground to form a *connected wall* matching the GT border skeleton.
  2. seam-core penalty (`seam_core_penalty`): on GT inter-fragment seam voxels
     (border classes), pushes total border probability up — i.e. punishes the
     specific "core fills the seam" failure that creates the bridge.

Border classes for 7-class pelvic border-core are {2,4,6} (sacrum/LH/RH seam);
core classes are {1,3,5}. See CLASS_TO_ANATOMY in scripts.fcirl_loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BORDER_CLASSES = (2, 4, 6)
CORE_CLASSES = (1, 3, 5)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """3D soft morphological erosion via min-pool (== -maxpool(-x))."""
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """3D soft morphological dilation via max-pool."""
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft skeleton (Shit et al., clDice, CVPR 2021).

    x : (B,1,D,H,W) foreground probability in [0,1]. Returns same shape soft skeleton.
    Iteratively erodes and accumulates the residual that opening removes (the medial
    ridge). `iters` should cover the max expected half-thickness of the structure.
    """
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        opened = _soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_border_loss(seg_logits: torch.Tensor,
                       label: torch.Tensor,
                       border_classes=BORDER_CLASSES,
                       iters: int = 10,
                       smooth: float = 1.0) -> torch.Tensor:
    """clDice loss on the border foreground.

    Rewards the predicted border's centerline overlapping the GT border centerline
    (Tprec) and vice-versa (Tsens) -> forces a *continuous separating wall* rather
    than a band that per-voxel CE leaves perforated.

    seg_logits : (B,C,D,H,W) raw logits.
    label      : (B,1,D,H,W) or (B,D,H,W) int GT class map.
    returns    : scalar in [0,1] (1 - clDice).
    """
    if label.dim() == 5:
        label = label[:, 0]
    prob = F.softmax(seg_logits, dim=1)
    p_border = sum(prob[:, c] for c in border_classes).unsqueeze(1)        # (B,1,D,H,W)
    g_border = torch.zeros_like(label, dtype=prob.dtype)
    for c in border_classes:
        g_border = g_border + (label == c).to(prob.dtype)
    g_border = g_border.clamp(0, 1).unsqueeze(1)

    sp = soft_skeletonize(p_border, iters)
    sg = soft_skeletonize(g_border, iters)
    tprec = (torch.sum(sp * g_border) + smooth) / (torch.sum(sp) + smooth)
    tsens = (torch.sum(sg * p_border) + smooth) / (torch.sum(sg) + smooth)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens)
    return 1.0 - cl_dice


def seam_core_penalty(seg_logits: torch.Tensor,
                      label: torch.Tensor,
                      border_classes=BORDER_CLASSES,
                      eps: float = 1e-6) -> torch.Tensor:
    """On GT seam (border-class) voxels, push total border probability up.

    Directly punishes the dominant merge mechanism (model predicts core, not border,
    on ~42% of true seam voxels -> bridge). L = mean over seam voxels of
    -log(sum_softmax(border_classes)). Returns 0 if there are no seam voxels.

    seg_logits : (B,C,D,H,W) raw logits.
    label      : (B,1,D,H,W) or (B,D,H,W) int GT class map.
    """
    if label.dim() == 5:
        label = label[:, 0]
    prob = F.softmax(seg_logits, dim=1)
    p_border = sum(prob[:, c] for c in border_classes)                    # (B,D,H,W)
    seam = torch.zeros_like(label, dtype=torch.bool)
    for c in border_classes:
        seam |= (label == c)
    if not bool(seam.any()):
        return seg_logits.sum() * 0.0
    m = seam.to(prob.dtype)
    nll = -torch.log(p_border.clamp_min(eps))
    return (nll * m).sum() / (m.sum() + eps)
