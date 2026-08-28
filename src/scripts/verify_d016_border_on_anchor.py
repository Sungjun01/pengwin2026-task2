"""Step 1 검증: D016이 놓친 seam의 anchor(밝은 부분)에서 border를 예측하나?

면-완성 decode 가능성의 핵심 게이트:
  Case A (decode 가능): D016이 anchor에서 border(class 2)를 *그리는데* 흐린 구간서 끊김
    → 끊긴 면을 이으면 분리됨. 재학습 0.
  Case B (재학습 필요): D016이 seam 전체를 core(class 1)로 예측 = anchor에서도 border 0
    → 이을 게 없음. ViT/Mamba long-range로 *border 예측 자체를* 재학습해야.

각 놓친 seam(GT 병합쌍) 표면에서:
  - 전체 seam voxel 중 D016 border(sac=2) 비율
  - anchor sub-portion(대조>=80HU) 중 D016 border 비율  ← 핵심
  - seam→가장가까운 D016 border voxel 거리(near=2voxel 내) 비율
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, binary_dilation, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
SAC_CORE, SAC_BORD = 1, 2
CASES = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac(lbl):
    return {int(v): (lbl == v) for v in np.unique(lbl) if 1 <= v <= 50}


def main():
    on_anchor = []      # anchor seam voxel 중 D016 border 비율
    on_all = []         # 전체 seam voxel 중 border 비율
    near_anchor = []    # anchor seam voxel 중 D016 border 2voxel내 비율
    n_seam = 0
    for c in CASES:
        cid = f"PENGWIN_{c}"
        pp = Path(OOF) / f"{cid}.nii.gz"; gp = Path(RAW) / c / "label.mha"; ip = Path(RAW) / c / "image.mha"
        if not (pp.exists() and gp.exists() and ip.exists()):
            continue
        pim, sem = lps(pp); sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(gp); _, ct = lps(ip); ct = ct.astype(np.float32)
        inst = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        inst = remove_small_components(inst.astype(np.uint16), spz, 1000.0)
        gi = sac(gt); pi = sac(inst)
        if len(gi) < 2:
            continue
        d016_border = (sem == SAC_BORD)
        border_dist = distance_transform_edt(~d016_border) if d016_border.any() else np.full(sem.shape, 99.0)
        for pid, pmask in pi.items():
            cov = [g for g, gm in gi.items() if (pmask & gm).sum() > 0.2 * gm.sum()]
            for i in range(len(cov)):
                for j in range(i + 1, len(cov)):
                    A, B = gi[cov[i]], gi[cov[j]]
                    seam = (binary_dilation(A, ST, 1) & B) | (binary_dilation(B, ST, 1) & A)
                    if seam.sum() < 5:
                        continue
                    n_seam += 1
                    bone = ct[(A | B)]; bm = float(np.median(bone[bone >= 0])) if (bone >= 0).any() else 200.0
                    sidx = np.argwhere(seam)
                    contrast = bm - ct[seam]
                    is_border = d016_border[seam]
                    dist = border_dist[seam]
                    on_all.append(float(is_border.mean()))
                    anc = contrast >= 80
                    if anc.sum() >= 1:
                        on_anchor.append(float(is_border[anc].mean()))
                        near_anchor.append(float((dist[anc] <= 2).mean()))
    print(f"=== Step 1: D016 border on missed-seam anchors (n_seam={n_seam}) ===")
    if on_anchor:
        print(f"anchor(>=80HU) seam voxel 중 D016 border(class2): 중앙값 {100*np.median(on_anchor):.0f}%  평균 {100*np.mean(on_anchor):.0f}%")
        print(f"anchor seam voxel 중 D016 border 2voxel내(near): 중앙값 {100*np.median(near_anchor):.0f}%  평균 {100*np.mean(near_anchor):.0f}%")
        print(f"전체 seam voxel 중 D016 border: 중앙값 {100*np.median(on_all):.0f}%")
        med_near = np.median(near_anchor)
        print(f"\n판정:")
        if med_near >= 0.3:
            print(f"  → Case A: D016이 anchor 근처에 border를 *그림*({100*med_near:.0f}% near) but 끊김 "
                  f"→ 면-완성 decode 유효 (재학습 0)")
        else:
            print(f"  → Case B: D016이 anchor에서도 border 거의 안 그음({100*med_near:.0f}% near) "
                  f"→ decode론 이을 게 없음. ViT/Mamba long-range로 border 예측 자체를 재학습 필요")
    print("STEP1_DONE")


if __name__ == "__main__":
    main()
