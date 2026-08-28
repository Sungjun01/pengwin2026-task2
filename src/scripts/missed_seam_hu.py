"""B: D016 가 놓친 seam vs 찾은 seam 의 HU 대조 비교 (영상한계 vs 방법한계 결정).

각 사크럼 골절 케이스에서:
  - D016 legacy decode → pred instance
  - 각 GT 사크럼 fragment 의 dominant pred label 계산
  - 인접 GT fragment 쌍이 같은 pred label 로 뭉치면 = 그 seam 'MISSED', 다르면 'FOUND'
  - 각 seam 의 GT 경계 voxel HU vs 두 fragment 내부 HU 대조 측정
집계: MISSED seam 대조 분포 vs FOUND seam 대조 분포.
  MISSED 가 체계적으로 더 낮으면 → 영상한계(신호 없음, 아무도 못함)
  MISSED ~ FOUND 면 → 방법한계(모델이 봤어야 함 → SHARC 재학습 EV)

용법: python -m scripts.missed_seam_hu --pred_dir /tmp/pred_d016_frac
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
MIN_FRAG_VOX = 200  # 너무 작은 GT 조각 무시


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/tmp/pred_d016_frac")
    args = ap.parse_args()
    preds = sorted(Path(args.pred_dir).glob("PENGWIN_*.nii.gz"))
    print(f"[missed_seam_hu] {len(preds)} cases")

    found_contrast, missed_contrast = [], []
    found_seamHU, missed_seamHU = [], []
    n_found = n_missed = 0
    for p in preds:
        cid = p.name.replace("PENGWIN_", "").replace(".nii.gz", "")
        pim, sem_full = lps(p)
        sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt_full = lps(Path(RAW) / cid / "label.mha")
        _, ct_full = lps(Path(RAW) / cid / "image.mha")
        ct_full = ct_full.astype(np.float32)
        # 사크럼 영역만 crop (decode + HU 속도)
        fgc = (sem_full > 0) | ((gt_full >= 1) & (gt_full <= 50))
        if not fgc.any():
            continue
        zz, yy, xx = np.where(fgc); pad = 6
        slz = (slice(max(0, zz.min()-pad), zz.max()+pad+1),
               slice(max(0, yy.min()-pad), yy.max()+pad+1),
               slice(max(0, xx.min()-pad), xx.max()+pad+1))
        sem = sem_full[slz]; gt = gt_full[slz]; ct = ct_full[slz]
        pred = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)

        sac = [int(v) for v in np.unique(gt) if 1 <= v <= 50 and (gt == v).sum() >= MIN_FRAG_VOX]
        if len(sac) < 2:
            continue
        # 각 GT 사크럼 조각의 dominant pred label
        dom = {}
        for v in sac:
            pv = pred[gt == v]
            pv = pv[pv > 0]
            dom[v] = int(np.bincount(pv).argmax()) if pv.size else 0
        # 인접 쌍 찾기 + seam HU
        for i in range(len(sac)):
            for j in range(i + 1, len(sac)):
                a, b = sac[i], sac[j]
                ma, mb = (gt == a), (gt == b)
                # 인접? a 팽창이 b 와 겹치면
                contact = binary_dilation(ma, ST, iterations=1) & mb
                if contact.sum() < 10:
                    continue
                # seam 경계 voxel (a 와 b 양쪽 1-voxel 띠)
                seam = (binary_dilation(ma, ST, 1) & mb) | (binary_dilation(mb, ST, 1) & ma)
                seam_hu = float(np.median(ct[seam]))
                inner = float(np.median(np.concatenate([
                    ct[ma & ~binary_dilation(seam, ST, 3)][:5000] if (ma & ~binary_dilation(seam, ST, 3)).any() else ct[ma],
                    ct[mb & ~binary_dilation(seam, ST, 3)][:5000] if (mb & ~binary_dilation(seam, ST, 3)).any() else ct[mb],
                ])))
                contrast = inner - seam_hu  # 양수 = seam 저밀도
                merged = (dom[a] == dom[b] and dom[a] != 0)
                if merged:
                    n_missed += 1; missed_contrast.append(contrast); missed_seamHU.append(seam_hu)
                else:
                    n_found += 1; found_contrast.append(contrast); found_seamHU.append(seam_hu)
        print(f"  {cid}: GT사크럼 {len(sac)}, found seams {n_found}, missed {n_missed}", flush=True)

    fc, mc = np.array(found_contrast), np.array(missed_contrast)
    print("\n" + "=" * 60)
    print(f"FOUND  seams (D016가 분리): n={len(fc)}  대조 median={np.median(fc) if len(fc) else float('nan'):.0f} HU  (mean {fc.mean() if len(fc) else float('nan'):.0f})")
    print(f"MISSED seams (D016가 병합): n={len(mc)}  대조 median={np.median(mc) if len(mc) else float('nan'):.0f} HU  (mean {mc.mean() if len(mc) else float('nan'):.0f})")
    if len(fc) and len(mc):
        print(f"\n차이 (found - missed 대조): {np.median(fc) - np.median(mc):.0f} HU")
        print("판정:")
        print("  MISSED 대조가 FOUND보다 뚜렷이 낮음 → 영상한계(놓친 seam은 신호 약함) → 사크럼 추격 중단")
        print("  MISSED ~ FOUND → 방법한계(모델이 봤어야 함) → SHARC 고해상도 재학습 EV 있음")
    print("MISSED_SEAM_DONE")


if __name__ == "__main__":
    main()
