"""BoRT (Skea-Topo, IJCAI'24, arXiv:2404.18539) adapted to the 7-class pelvic border-core.
TRAIN-ONLY.

Genuinely-new lever vs the already-tried BorderTopo (clDice + flat seam-core): the v18
BORDER_WEIGHT=40 weights EVERY seam voxel equally, so a thin iso-density seam (where two
fragments nearly touch — exactly where the model merges) gets no more gradient than a fat
obvious one. BoRT's skeleton-aware weight upweights seams INVERSELY to their local thickness,
so the thinnest seams get the strongest push toward 'border'.

Two terms:
  L_skeaw : weighted CE over GT seam voxels, weight = 1 + w0*(Wmax - thickness)/Wmax, where
            thickness ~ 2 * (distance to nearest fragment core) at the seam (on-the-fly EDT
            from GT per bone). Thin seam -> big weight.
  L_topo  : on the GT seam SKELETON (the centerline the model must keep as a wall), push total
            border probability up (reuses the clDice soft skeleton). Light.

Weights/skeletons are non-differentiable and only select WHERE/HOW-MUCH the base CE applies;
gradient flows through CE only. Inference graph unchanged -> T4 cost identical to v18.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from scripts.topology_border_loss import soft_skeletonize

BONES = ((1, 2), (3, 4), (5, 6))  # (core_class, seam/border_class)


def _seam_narrow_weight(gt_np, w0, wmax_vox):
    """numpy single volume -> per-voxel weight (1.0 base; thin seams up to 1+w0)."""
    W = np.ones(gt_np.shape, np.float32)
    for core_c, border_c in BONES:
        seam = gt_np == border_c
        core = gt_np == core_c
        if not seam.any() or not core.any():
            continue
        d_core = ndi.distance_transform_edt(~core)        # voxel dist to nearest core
        thick = 2.0 * d_core                              # ~ local inter-fragment seam width
        narrow = np.clip((wmax_vox - thick) / max(wmax_vox, 1e-3), 0.0, 1.0)
        W[seam] = 1.0 + w0 * narrow[seam]
    return W


def bort_loss(seg_logits, label, w0=10.0, lambda_topo=0.1, wmax_mm=8.0,
              spacing_mm=1.0, skel_iters=10):
    """Returns (total, l_skeaw.detach(), l_topo.detach())."""
    if label.dim() == 5:
        label = label[:, 0]
    label = label.long()
    logits = seg_logits.float()
    B = logits.shape[0]
    wmax_vox = max(wmax_mm / max(spacing_mm, 1e-3), 1.0)

    with torch.no_grad():
        gt_np = label.cpu().numpy()
        Wnp = np.stack([_seam_narrow_weight(gt_np[b], w0, wmax_vox) for b in range(B)])
    W = torch.as_tensor(Wnp, device=logits.device)

    seam = torch.zeros_like(label, dtype=torch.bool)
    for _c, bc in BONES:
        seam |= label == bc

    ce = F.cross_entropy(logits, label, reduction="none")  # (B,Z,Y,X)
    if bool(seam.any()):
        wseam = W * seam
        l_skeaw = (wseam * ce).sum() / wseam.sum().clamp_min(1.0)
    else:
        l_skeaw = logits.sum() * 0.0

    if lambda_topo > 0 and bool(seam.any()):
        prob = F.softmax(logits, dim=1)
        p_border = sum(prob[:, bc] for _c, bc in BONES).unsqueeze(1)  # (B,1,Z,Y,X)
        g_seam = seam.unsqueeze(1).to(prob.dtype)
        sk = soft_skeletonize(g_seam, skel_iters)                    # (B,1,Z,Y,X)
        nll = -(p_border.clamp_min(1e-6)).log()
        l_topo = (nll * sk).sum() / sk.sum().clamp_min(1.0)
    else:
        l_topo = logits.sum() * 0.0

    total = l_skeaw + lambda_topo * l_topo
    return total, l_skeaw.detach(), l_topo.detach()
