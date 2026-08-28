"""골절-위치 atlas (heatmap) — 깨끗한 smoothed 버전, repo에 영구 저장.

build_fracture_atlas.py(raw 5mm 히스토그램)를 계승하되:
  - 170 pelvic GT 의 inter-fragment seam voxel 을 사크럼 centroid 기준 offset(mm)으로 누적
  - 3D 가우시안 smooth (raw bin 노이즈 제거 → soft prior; report 36 결론: GMM 대신 smoothed atlas)
  - max 정규화 → [0,1]
  - assets/fracture_atlas.npz 로 저장 (atlas + R/BIN/N) — 손실/추론이 안정 경로에서 로드

좌표: cen=center_of_mass(sacrum) (zyx voxel), off_mm=(coord-cen)*spacing_zyx, idx=((off_mm+R)/BIN).

용법: python -m scripts.build_fracture_atlas_v2
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

R, BIN, N = 150.0, 5.0, 60   # ±150mm, 5mm bins, 60^3
SMOOTH_SIGMA_BINS = 1.5
ANAT = [(1, 50), (51, 100), (101, 150)]
GTD = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance"
OUT = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/assets/fracture_atlas.npz"


def main():
    hist = np.zeros((N, N, N), np.float64)
    ncase = 0
    for f in sorted(glob.glob(GTD + "/*.nii.gz")):
        img = sitk.ReadImage(f); arr = sitk.GetArrayFromImage(img)
        sx, sy, sz = img.GetSpacing(); sp = np.array([sz, sy, sx])  # zyx mm
        sac = (arr >= 1) & (arr <= 50)
        if not sac.any():
            continue
        cen = np.array(ndi.center_of_mass(sac))  # zyx voxel
        seam = np.zeros(arr.shape, bool)
        for lo, hi in ANAT:
            a = np.where((arr >= lo) & (arr <= hi), arr, 0)
            for shift in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nb = np.roll(a, shift, axis=(0, 1, 2)); seam |= (a > 0) & (nb > 0) & (a != nb)
        if not seam.any():
            continue
        zz, yy, xx = np.where(seam)
        off_mm = (np.stack([zz, yy, xx], 1).astype(np.float64) - cen) * sp
        idx = ((off_mm + R) / BIN).astype(int)
        ok = np.all((idx >= 0) & (idx < N), axis=1); idx = idx[ok]
        np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        ncase += 1
    hp = hist / max(ncase, 1)
    # peakiness (raw)
    flat = hp[hp > 0]
    top10 = np.percentile(flat, 90)
    print(f"cases={ncase}, occupied bins={len(flat)}, max/mean={hp.max()/flat.mean():.1f}, "
          f"top-10% mass={100*hp[hp>=top10].sum()/hp.sum():.1f}% (uniform=10%)")
    # smooth → soft prior
    sm = ndi.gaussian_filter(hp, SMOOTH_SIGMA_BINS)
    sm = sm / sm.max()  # [0,1]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, atlas=sm.astype(np.float32), R=R, BIN=BIN, N=N,
                        smooth_sigma_bins=SMOOTH_SIGMA_BINS)
    print(f"saved smoothed atlas -> {OUT}  (shape {sm.shape}, max=1.0, "
          f"mean={sm.mean():.3f}, frac>0.3={100*(sm>0.3).mean():.1f}%)")
    print("ATLAS_V2_DONE")


if __name__ == "__main__":
    main()
