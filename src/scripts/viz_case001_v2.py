"""case 001 상세 시각화 v2 — 큰 패널 + 작은 슬리버(IoU 취약) 강조 + 부피 범례 + v18/v24 비교.

001: v18=0.75 vs 1등=0.833 (둘 다 9조각). 작은 슬리버가 경계-IoU 격차 원인.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"
CT = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/progress/dataset_validation/001_input.mha"
P18 = "/tmp/real001_v18/out/images/peripelvic-fracture-ct-segmentation/real001.mha"
P24 = "/tmp/real001_v24/out/images/peripelvic-fracture-ct-segmentation/real001.mha"


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im), im.GetSpacing()


def relbl(a):
    """anatomy 순서로 1.. relabel + (id->(anat,voxels)) 매핑."""
    out = np.zeros_like(a, dtype=np.int32); info = {}; nxt = 1
    for anat, (lo, hi) in [("sac", (1, 50)), ("LH", (51, 100)), ("RH", (101, 150))]:
        for v in [v for v in np.unique(a) if lo <= v <= hi]:
            out[a == v] = nxt; info[nxt] = (anat, int((a == v).sum())); nxt += 1
    return out, info


def main():
    _, ct, sp = lps(CT); ct = ct.astype(np.float32)
    vox_mm3 = float(np.prod(sp))
    _, p18, _ = lps(P18); _, p24, _ = lps(P24)
    lbl, info = relbl(p18)
    s24, _ = relbl(p24)
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    fg = lbl > 0; zz, yy, xx = np.where(fg)
    zc = [int(np.percentile(zz, p)) for p in (25, 45, 65)]
    ymid = int(np.median(yy)); xmid = int(np.median(xx))

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.7])
    panels = [(0, 0, "axial z=%d" % zc[0], ct[zc[0]], lbl[zc[0]]),
              (0, 1, "axial z=%d" % zc[1], ct[zc[1]], lbl[zc[1]]),
              (0, 2, "axial z=%d" % zc[2], ct[zc[2]], lbl[zc[2]]),
              (1, 0, "coronal", ct[:, ymid], lbl[:, ymid]),
              (1, 1, "sagittal", ct[:, :, xmid], lbl[:, :, xmid])]
    for r, c, t, bg, ov in panels:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(bg, cmap="gray", vmin=-200, vmax=1100, aspect="auto")
        ax.imshow(np.ma.masked_where(ov == 0, ov), cmap=cmap, alpha=0.5, vmin=1, vmax=20, aspect="auto")
        ax.set_title(f"001 {t}", fontsize=11); ax.axis("off")
    # v18 vs v24 사크럼 비교 (surfcomp가 001 사크럼 바꿨나)
    sac18 = np.where((lbl >= 1) & (lbl <= 6), lbl, 0)
    sac24 = np.where((s24 >= 1) & (s24 <= 6), s24, 0)
    zs = max(range(sac18.shape[0]), key=lambda z: (sac18[z] > 0).sum())
    for col, (tag, arr) in enumerate([("v18 sacrum", sac18), ("v24 sacrum", sac24)]):
        ax = fig.add_subplot(gs[1, 2] if col == 0 else gs[1, 3])
        ax.imshow(ct[zs], cmap="gray", vmin=-200, vmax=1100)
        ax.imshow(np.ma.masked_where(arr[zs] == 0, arr[zs]), cmap=cmap, alpha=0.6, vmin=1, vmax=20)
        ax.set_title(f"001 {tag} (z={zs})", fontsize=10); ax.axis("off")

    # 범례: fragment 부피 + IoU 취약/제거 표시
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    lines = ["fragment 부피 (mm³):", ""]
    for fid in sorted(info):
        anat, nv = info[fid]; mm3 = nv * vox_mm3
        flag = "  ← <1cm³ 채점제거" if mm3 < 1000 else ("  ← 작음(IoU취약)" if mm3 < 6000 else "")
        col = plt.cm.tab20((fid - 1) / 19.0)
        lines.append(f"#{fid} {anat}: {mm3:6.0f}{flag}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace",
            transform=ax.transAxes)
    ax.text(0.0, 0.18, "큰 몸통 3개 + 작은 슬리버들.\n작은 조각 = 경계 IoU 취약\n→ 001 격차(0.75 vs 0.833)의 원인",
            va="top", ha="left", fontsize=11, color="#d62728", transform=ax.transAxes)

    plt.suptitle("case 001 (pelvic ring fracture) — CT + v18 prediction + v18/v24 sacrum compare. "
                 "9 fragments; small slivers drive the boundary-IoU gap vs 1st place.", fontsize=13)
    plt.tight_layout()
    out = "figs/case001_detail_v2.png"
    plt.savefig(out, dpi=115, bbox_inches="tight"); plt.close()
    print("saved", out)
    # v18 vs v24 사크럼 변화 정량
    print(f"\nv18 사크럼 voxel: {(sac18>0).sum()}, v24 사크럼 voxel: {(sac24>0).sum()}, "
          f"차이 voxel: {int(np.abs((sac18>0).astype(int)-(sac24>0).astype(int)).sum())}")
    print("VIZ_001_V2_DONE")


if __name__ == "__main__":
    main()
