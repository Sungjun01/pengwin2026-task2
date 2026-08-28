"""가우시안 Soft Skeleton-Recall 타겟 시각화 (학습 전 sanity check).

GT 골절면 S → EDT D(x) → 가우시안 soft label G(x)=exp(-D²/2σ²).
σ 별로 soft band 두께가 CT 위에 어떻게 씌워지는지 → σ 선택 근거.

사크럼 골절 케이스(097, GT 7조각)에서 사크럼 골절면만.
용법: python -m scripts.viz_gaussian_target --case 097
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fsdf_experiment import fracture_surface

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
SIGMAS = [0.75, 1.0, 1.5, 2.0]


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="097")
    ap.add_argument("--axis", default="axial", choices=["axial", "coronal", "sagittal"])
    args = ap.parse_args()
    c = args.case
    gimg, gt = lps(Path(RAW) / c / "label.mha")
    _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
    sp = gimg.GetSpacing(); samp = sp[::-1]  # (z,y,x) mm/voxel

    sac = ((gt >= 1) & (gt <= 50))
    # 사크럼 골절면 S (인접 다른 fragment 경계)
    S = fracture_surface(gt, sac)
    if not S.any():
        print("골절면 없음"); return
    # EDT (mm 단위), 가우시안 → voxel 단위 σ 로 변환위해 voxel EDT 사용
    D_vox = distance_transform_edt(~S)  # voxel 거리
    print(f"case {c}: 사크럼 골절면 voxel수={int(S.sum())}, GT 사크럼 {len([v for v in np.unique(gt) if 1<=v<=50])}조각")

    # 골절면이 가장 많은 슬라이스
    cnt = [(int(S[z].sum()), z) for z in range(S.shape[0])]
    z = max(cnt)[1]
    cts = ct[z]; sacs = sac[z]

    fig, axes = plt.subplots(1, len(SIGMAS) + 1, figsize=(4.2 * (len(SIGMAS) + 1), 5))
    # (0) CT + 사크럼 윤곽 + 골절선
    axes[0].imshow(cts, cmap="gray", vmin=-200, vmax=900)
    axes[0].imshow(np.ma.masked_where(~sacs, sacs), cmap="Blues", alpha=0.25)
    axes[0].imshow(np.ma.masked_where(~S[z], S[z]), cmap="autumn", alpha=0.9)
    axes[0].set_title(f"(0) CT + 사크럼(파랑) + GT골절선(빨강)\ncase {c}, z={z}"); axes[0].axis("off")
    # σ별 가우시안 타겟
    for i, sig in enumerate(SIGMAS, 1):
        G = np.exp(-(D_vox[z] ** 2) / (2 * sig ** 2))
        G[~sacs] = 0  # 사크럼 내부만
        axes[i].imshow(cts, cmap="gray", vmin=-200, vmax=900)
        im = axes[i].imshow(np.ma.masked_where(G < 0.05, G), cmap="hot", alpha=0.75, vmin=0, vmax=1)
        # G=0.5 등고선 (band 두께 가늠)
        axes[i].contour(G, levels=[0.5], colors="cyan", linewidths=0.8)
        axes[i].set_title(f"(σ={sig}) Gaussian soft target\ncyan=확률0.5 경계"); axes[i].axis("off")
    plt.suptitle(f"Gaussian Soft Skeleton-Recall 타겟 — σ별 soft band 두께 (G=exp(-D²/2σ²))", fontsize=13)
    plt.tight_layout()
    out = f"figs/gaussian_target_{c}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)
    # σ별 band 통계
    for sig in SIGMAS:
        G = np.exp(-(D_vox ** 2) / (2 * sig ** 2)); G[~sac] = 0
        frac05 = float((G[sac] >= 0.5).mean()) * 100
        frac01 = float((G[sac] >= 0.1).mean()) * 100
        print(f"  σ={sig}: 사크럼 중 G≥0.5 비중={frac05:.1f}%, G≥0.1={frac01:.1f}% (band 두께 가늠)")


if __name__ == "__main__":
    main()
