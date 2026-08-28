"""Diagnose HOW held-out predictions fail: CT | GT fragments | pred-core CC | pred-border.

For each target (case, anatomy), one row of 4 axial panels on the best slice:
  1. CT (bone window)
  2. CT + GT instance fragments (each color = one true fragment)
  3. CT + predicted CORE connected-components (each color = one recovered piece)
  4. CT + predicted BORDER/seam (red) over faint core
=> over-seg via "border splits a solid bone" (hallucinated seam) vs "core has gaps"
   becomes visible; under-seg via "no border where GT has a fracture" becomes visible.

용법: python3 scripts/viz_heldout_diagnosis.py 010:sac 025:LH 011:RH
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label

ST = generate_binary_structure(3, 3)
PRED = os.environ.get("VIZ_PRED", "predictions/heldout_best/out")
CT = os.environ.get("VIZ_CT", "predictions/heldout_ep450/in")
GT = os.environ.get("VIZ_GT", "predictions/gt_instance")
ANAT = {"sac": (1, 2, (1, 50)), "LH": (3, 4, (51, 100)), "RH": (5, 6, (101, 150))}
ABS_MM = 1000.0


def bone_window(ct):
    return np.clip((ct + 200.0) / 1400.0, 0, 1)


def relabel_gt(g, lo, hi, vmm):
    out = np.zeros_like(g, dtype=np.int32); k = 0
    for fid in range(lo, hi + 1):
        m = g == fid
        if m.sum() * vmm >= ABS_MM:
            k += 1; out[m] = k
    return out, k


def cc_relabel(mask, vmm):
    lab, n = cc_label(mask, structure=ST)
    out = np.zeros_like(lab, dtype=np.int32); k = 0
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() * vmm >= ABS_MM:
            k += 1; out[m] = k
    return out, k


def main(targets):
    fig, axes = plt.subplots(len(targets), 4, figsize=(20, 5 * len(targets)))
    if len(targets) == 1:
        axes = axes[None, :]
    for r, tgt in enumerate(targets):
        cid, an = tgt.split(":")
        core_c, bord_c, (lo, hi) = ANAT[an]
        pimg = sitk.ReadImage(f"{PRED}/PENGWIN_{cid}.nii.gz")
        parr = sitk.GetArrayFromImage(pimg); vmm = float(np.prod(pimg.GetSpacing()))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(f"{CT}/PENGWIN_{cid}_0000.nii.gz")).astype(np.float32)
        g = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT}/PENGWIN_{cid}.nii.gz"))

        gt_v, gt_k = relabel_gt(g, lo, hi, vmm)
        core_v, core_k = cc_relabel(parr == core_c, vmm)
        bord = (parr == bord_c)

        region = (gt_v > 0) | (core_v > 0)
        z = int(np.argmax([(region[zz]).sum() for zz in range(region.shape[0])]))
        zz, yy, xx = np.where(region)
        y0, y1, x0, x1 = yy.min() - 8, yy.max() + 8, xx.min() - 8, xx.max() + 8
        y0, x0 = max(0, y0), max(0, x0)
        sl = (z, slice(y0, y1), slice(x0, x1))
        ctw = bone_window(ct[sl])

        for c in range(4):
            axes[r, c].imshow(ctw, cmap="gray"); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_title(f"PENGWIN_{cid} {an}  CT z={z}", fontsize=10)
        gm = gt_v[sl]
        axes[r, 1].imshow(np.ma.masked_where(gm == 0, gm), cmap="tab20", alpha=0.65, vmin=1, vmax=20, interpolation="nearest")
        axes[r, 1].set_title(f"GT fragments = {gt_k}", fontsize=10)
        cm = core_v[sl]
        axes[r, 2].imshow(np.ma.masked_where(cm == 0, cm), cmap="tab20", alpha=0.65, vmin=1, vmax=20, interpolation="nearest")
        axes[r, 2].set_title(f"pred CORE CC = {core_k}", fontsize=10)
        axes[r, 3].imshow(np.ma.masked_where(~(core_v[sl] > 0), core_v[sl] > 0), cmap=ListedColormap(["yellow"]), alpha=0.25)
        axes[r, 3].imshow(np.ma.masked_where(~bord[sl], bord[sl]), cmap=ListedColormap(["red"]), alpha=0.9)
        axes[r, 3].set_title("pred BORDER(red) over core(yellow)", fontsize=10)
    fig.suptitle(os.environ.get("VIZ_TITLE", "Held-out failure diagnosis: GT vs predicted core/border"), fontsize=13, y=1.0)
    fig.tight_layout()
    out = os.environ.get("VIZ_OUT", "viz/heldout_diagnosis.png")
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  (targets: {', '.join(targets)})")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["010:sac"])
