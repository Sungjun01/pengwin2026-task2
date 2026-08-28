"""Supervoxel critical-component topology loss (arXiv:2501.01022) for pelvic border-core.

TRAIN-ONLY. Attacks the v18 "confident-core" merge failure: at a true fracture seam the
network confidently predicts CORE (not border), so the 26-conn decode bridges two
fragments. Per-voxel CE is blind to this — a thin bridge of a few voxels is catastrophic
for instances yet nearly free for CE.

This loss detects, in the network's argmax prediction, false-MERGE BRIDGES: false-positive
connected components that join two distinct GT fragment instances. It then applies the base
cross-entropy to each such component with PER-COMPONENT averaging, so a thin seam bridge is
penalized as heavily as a thick one (thickness-independent — the whole point). Symmetrically
it penalizes false-SPLIT gaps (FN components that separate two predicted pieces of ONE GT
fragment), with a smaller weight.

Fragment instances are derived on-the-fly from GT *core* connected components per bone
(sacrum 1/2, leftHip 3/4, rightHip 5/6) — the border class is the separating wall — so NO
instance-label dataset is needed (same on-the-fly-from-GT idea the F-CIRL Aseg trainer uses).

CC labelling and the criticality test are non-differentiable and only build per-voxel
component-id maps; gradient flows solely through the masked cross-entropy on the seg logits,
exactly as in the paper (the supervoxels select WHERE the base loss is applied).

Inference graph is untouched (this only shapes the training loss) -> T4 cost identical to v18.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

# (core_class, border_class) for the 7-class pelvic border-core label scheme
BONES = ((1, 2), (3, 4), (5, 6))
_STRUCT = ndi.generate_binary_structure(3, 1)  # 6-connectivity


def _expand_slice(sl, pad, shape):
    return tuple(
        slice(max(0, s.start - pad), min(n, s.stop + pad)) for s, n in zip(sl, shape)
    )


def _build_critical_maps(pred, gt, min_size: int = 2):
    """numpy, single volume. Returns (merge_id, split_id) int32 maps (0 = none).

    merge_id: voxels of FP components that bridge >=2 GT fragment instances (false merge).
    split_id: voxels of FN components that separate >=2 predicted pieces of one GT instance.
    Component ids are unique within this call.
    """
    shape = gt.shape
    merge_id = np.zeros(shape, np.int32)
    split_id = np.zeros(shape, np.int32)
    next_m, next_s = 1, 1
    for core_c, _border_c in BONES:
        gt_core = gt == core_c
        if not gt_core.any():
            continue
        gt_inst, n_gt = ndi.label(gt_core, _STRUCT)
        pred_core = pred == core_c

        # ---- false-merge bridges: FP comp adjacent to >=2 GT instances ----
        fp = pred_core & ~gt_core
        if fp.any() and n_gt >= 2:
            fp_lab, _ = ndi.label(fp, _STRUCT)
            for cc, sl in enumerate(ndi.find_objects(fp_lab), start=1):
                if sl is None:
                    continue
                sl2 = _expand_slice(sl, 1, shape)
                comp = fp_lab[sl2] == cc
                if int(comp.sum()) < min_size:
                    continue
                dil = ndi.binary_dilation(comp, _STRUCT)
                gi_local = gt_inst[sl2]
                touched = np.unique(gi_local[dil & (gi_local > 0)])
                if touched.size >= 2:
                    sub = merge_id[sl2]
                    sub[comp] = next_m
                    merge_id[sl2] = sub
                    next_m += 1

        # ---- false-split gaps: FN comp between >=2 predicted pieces of ONE GT inst ----
        pred_inst, n_pi = ndi.label(pred_core, _STRUCT)
        fn = gt_core & ~pred_core
        if fn.any() and n_pi >= 2:
            fn_lab, _ = ndi.label(fn, _STRUCT)
            for cc, sl in enumerate(ndi.find_objects(fn_lab), start=1):
                if sl is None:
                    continue
                sl2 = _expand_slice(sl, 1, shape)
                comp = fn_lab[sl2] == cc
                if int(comp.sum()) < min_size:
                    continue
                gi = np.unique(gt_inst[sl2][comp])
                gi = gi[gi > 0]
                dil = ndi.binary_dilation(comp, _STRUCT)
                pi_local = pred_inst[sl2]
                touched_pred = np.unique(pi_local[dil & (pi_local > 0)])
                if gi.size == 1 and touched_pred.size >= 2:
                    sub = split_id[sl2]
                    sub[comp] = next_s
                    split_id[sl2] = sub
                    next_s += 1
    return merge_id, split_id


def _segment_mean_of_means(ce_flat: torch.Tensor, id_flat: torch.Tensor) -> torch.Tensor:
    """Mean over components of (mean CE within component). Thickness-independent."""
    mask = id_flat > 0
    if not bool(mask.any()):
        return ce_flat.new_zeros(())
    ids = id_flat[mask].long()
    ce = ce_flat[mask]
    n = int(ids.max().item()) + 1
    sums = ce.new_zeros(n).scatter_add_(0, ids, ce)
    cnts = ce.new_zeros(n).scatter_add_(0, ids, torch.ones_like(ce))
    present = cnts > 0
    return (sums[present] / cnts[present]).mean()


def supervoxel_critical_loss(
    seg_logits: torch.Tensor,
    label: torch.Tensor,
    w_merge: float = 1.0,
    w_split: float = 0.25,
    min_size: int = 2,
):
    """Supervoxel critical-component loss.

    seg_logits : (B, C, Z, Y, X) raw logits (C = 7 for pelvic border-core).
    label      : (B, 1, Z, Y, X) or (B, Z, Y, X) int GT class map.
    Returns (total_loss, l_merge.detach(), l_split.detach()).
    """
    if label.dim() == 5:
        label = label[:, 0]
    label = label.long()
    logits = seg_logits.float()
    B = logits.shape[0]

    with torch.no_grad():
        pred_np = logits.argmax(1).cpu().numpy()
        gt_np = label.cpu().numpy()
        merge_np = np.zeros_like(gt_np, dtype=np.int32)
        split_np = np.zeros_like(gt_np, dtype=np.int32)
        moff = soff = 0
        for b in range(B):
            m, s = _build_critical_maps(pred_np[b], gt_np[b], min_size=min_size)
            mp, sp = m > 0, s > 0
            if mp.any():
                m[mp] += moff
                moff = int(m.max())
            if sp.any():
                s[sp] += soff
                soff = int(s.max())
            merge_np[b] = m
            split_np[b] = s

    ce = F.cross_entropy(logits, label, reduction="none").reshape(-1)  # differentiable
    merge_id = torch.as_tensor(merge_np, device=ce.device).reshape(-1)
    split_id = torch.as_tensor(split_np, device=ce.device).reshape(-1)
    l_merge = _segment_mean_of_means(ce, merge_id)
    l_split = _segment_mean_of_means(ce, split_id)
    total = w_merge * l_merge + w_split * l_split
    return total, l_merge.detach(), l_split.detach()
