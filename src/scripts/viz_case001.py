"""case 001 상세 — CT + 우리 예측(v18) 조각 (공개 GT 없어 예측만). 001 경계-IoU 격차 진단용.

001 은 v18=0.75 vs 1등=0.833 (둘 다 9조각, merge_err=1, split_err=0.333) = 경계 IoU 품질 차이.
어디서 조각이 붙었나(merge), 경계가 거친가, 사크럼/힙 골절 패턴을 본다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.family"] = "DejaVu Sans"
CT = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/progress/dataset_validation/001_input.mha"
PRED = "/tmp/real001_v18/out/images/peripelvic-fracture-ct-segmentation/real001.mha"


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def relbl(a):
    """sac1-50 -> 1-, LH51-100 -> 7-, RH101-150 -> 13- 로 distinct 색."""
    out = np.zeros_like(a, dtype=np.int32)
    nxt = 1
    for lo, hi in [(1, 50), (51, 100), (101, 150)]:
        for v in [v for v in np.unique(a) if lo <= v <= hi]:
            out[a == v] = nxt; nxt += 1
    return out


def main():
    _, ct = lps(CT); ct = ct.astype(np.float32)
    _, pred = lps(PRED)
    lbl = relbl(pred)
    fg = lbl > 0
    if not fg.any():
        print("no pred"); return
    zz, yy, xx = np.where(fg)
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    # 3 axial (low/mid/high sacrum), 1 coronal, 1 sagittal
    zc = [int(np.percentile(zz, p)) for p in (30, 50, 70)]
    ymid = int(np.median(yy)); xmid = int(np.median(xx))
    fig, ax = plt.subplots(1, 5, figsize=(24, 5))
    views = [("axial z=%d" % zc[0], ct[zc[0]], lbl[zc[0]]),
             ("axial z=%d" % zc[1], ct[zc[1]], lbl[zc[1]]),
             ("axial z=%d" % zc[2], ct[zc[2]], lbl[zc[2]]),
             ("coronal", ct[:, ymid], lbl[:, ymid]),
             ("sagittal", ct[:, :, xmid], lbl[:, :, xmid])]
    for a, (t, bg, ov) in zip(ax, views):
        a.imshow(bg, cmap="gray", vmin=-200, vmax=1100, aspect="auto")
        a.imshow(np.ma.masked_where(ov == 0, ov), cmap=cmap, alpha=0.5, vmin=1, vmax=20, aspect="auto")
        a.set_title(f"001 {t}", fontsize=10); a.axis("off")
    nf = int(lbl.max())
    plt.suptitle(f"case 001 — CT + v18 prediction ({nf} fragments: sac+LH+RH). "
                 "v18=0.75 vs 1st=0.833 (same structure, boundary-IoU gap). color=fragment id.", fontsize=12)
    plt.tight_layout()
    out = "figs/case001_detail.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)
    # 조각별 부피 (작은 조각 = IoU 위험)
    print("\nfragment 부피(voxel) — 작은 조각이 경계-IoU에 취약:")
    for v in range(1, nf + 1):
        n = int((lbl == v).sum())
        anat = "sac" if v <= 6 else ("LH" if v <= 12 else "RH")  # 대략
        print(f"  frag {v:2d}: {n:8d} voxel")
    print("VIZ_001_DONE")


if __name__ == "__main__":
    main()
