"""OOF(held-out) missed-seam 게이트 — 오염됐던 train-set HU 게이트의 깨끗한 버전.

이전 HU 게이트는 fold_all 예측(=학습한 케이스)에서 쟀음 → 일반화엔 낙관적.
여기선 각 케이스를 *학습 안 한* fold 의 validation 예측(OOF)으로 재측정.

각 사크럼 골절 케이스:
  1. OOF 7-class 예측 → legacy decode → 사크럼 instance.
  2. GT 사크럼 instance 와 매칭. 인접 GT fragment 쌍 중 *예측이 한 덩어리로 병합*한 쌍 = 놓친 seam.
  3. 그 seam(두 GT fragment 경계면) 의 CT HU 대조 측정 → 회복가능(>=40HU) vs 영상한계(<40HU).

핵심 숫자: OOF 놓친 seam 중 회복가능 비율 r_oof.
  r_oof >= 25% → 재학습(Tversky recall) 정당. < 25% → 영상한계, 재학습 무의미.

용법: python -m scripts.oof_missed_seam_gate
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
GTD = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance"
OOF = "/tmp/oof_d016_sac"
LO, HI = 1, 50          # sacrum label range
CLEANUP = 1000.0


def lps_arr(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac_instances(lbl):
    """label map -> dict{id: bool mask} for sacrum fragments (1-50)."""
    out = {}
    for v in np.unique(lbl):
        if LO <= v <= HI:
            out[int(v)] = (lbl == v)
    return out


def boundary_hu(maskA, maskB, ct, spacing_zyx):
    """두 fragment 인접면(서로 1 voxel 이내) voxel 의 CT HU. seam 대조 측정용."""
    # A 를 1 dilate 해서 B 와 겹치는 영역 = 접촉면 근처
    dA = binary_dilation(maskA, ST, iterations=1)
    dB = binary_dilation(maskB, ST, iterations=1)
    seam = (dA & maskB) | (dB & maskA)
    if seam.sum() < 3:
        return None
    return ct[seam]


def main():
    frac = [l.strip() for l in open("/tmp/frac_sacrum_cases.txt") if l.strip()]
    all_missed_hu = []   # per missed-seam: median HU of the seam (lower => imaging limit)
    n_cases = 0
    per_case = []
    for c in frac:
        cid = f"PENGWIN_{c}"
        pp = Path(OOF) / f"{cid}.nii.gz"
        gp = Path(GTD) / f"{cid}.nii.gz"
        ip = Path(RAW) / c / "image.mha"
        if not (pp.exists() and gp.exists() and ip.exists()):
            continue
        n_cases += 1
        _, pred7 = lps_arr(pp)
        gimg, gt = lps_arr(gp)
        _, ct = lps_arr(ip); ct = ct.astype(np.float32)
        sp = gimg.GetSpacing()[::-1]
        vox = float(sp[0] * sp[1] * sp[2])
        # decode OOF 7-class -> instance (legacy), cleanup 1cm3
        inst = assemble_pelvic_instances_mode(
            pred7, vox, mode="legacy", spacing_zyx=tuple(sp), min_volume_mm3=500.0)
        inst = remove_small_components(inst, sp, CLEANUP)
        gi = sac_instances(gt)
        pi = sac_instances(inst)
        if len(gi) < 2:
            continue
        # 각 예측 사크럼 덩어리가 어떤 GT fragment 들을 덮나 (병합 검출)
        merged_pairs = 0
        case_missed = 0
        for pid, pmask in pi.items():
            covered = [gid for gid, gm in gi.items()
                       if (pmask & gm).sum() > 0.2 * gm.sum()]
            if len(covered) >= 2:
                # 이 예측 덩어리가 GT fragment 여러 개를 병합 = 놓친 seam(들)
                for i in range(len(covered)):
                    for j in range(i + 1, len(covered)):
                        gA, gB = gi[covered[i]], gi[covered[j]]
                        hu = boundary_hu(gA, gB, ct, sp)
                        if hu is None:
                            continue
                        # seam 대조: 인접 골(bone, >=200HU) 대비 seam median 의 차
                        bone = ct[(gA | gB)]
                        bone_med = float(np.median(bone[bone >= 0])) if (bone >= 0).any() else 200.0
                        seam_med = float(np.median(hu))
                        contrast = bone_med - seam_med  # 클수록 seam 이 어둡게 보임(회복쉬움)
                        all_missed_hu.append(contrast)
                        case_missed += 1
                merged_pairs += 1
        per_case.append((c, len(gi), len(pi), case_missed))

    print(f"=== OOF missed-seam 게이트 (held-out, 깨끗) — {n_cases} 케이스 ===")
    print(f"{'case':6s} {'GT조각':>6s} {'예측조각':>7s} {'놓친seam':>8s}")
    for c, ng, npd, nm in per_case:
        flag = "  <<병합" if nm > 0 else ""
        print(f"{c:6s} {ng:6d} {npd:7d} {nm:8d}{flag}")
    arr = np.array(all_missed_hu)
    print(f"\n총 놓친 seam 수 (OOF): {len(arr)}")
    if len(arr):
        b1 = int((arr < 0).sum()); b2 = int(((arr >= 0) & (arr < 40)).sum())
        b3 = int(((arr >= 40) & (arr < 80)).sum()); b4 = int((arr >= 80).sum())
        n = len(arr)
        print(f"  대조 <0 HU   : {b1} ({100*b1/n:.0f}%)  seam이 뼈보다 밝음")
        print(f"  대조 0~40 HU : {b2} ({100*b2/n:.0f}%)  영상한계")
        print(f"  대조 40~80 HU: {b3} ({100*b3/n:.0f}%)  회복가능")
        print(f"  대조 >=80 HU : {b4} ({100*b4/n:.0f}%)  강신호(method실패)")
        recover = b3 + b4
        print(f"\n영상한계(<40HU): {100*(b1+b2)/n:.0f}%  |  회복가능(>=40HU) r_oof={100*recover/n:.0f}%")
        print(f"판정: {'재학습 정당 (r_oof>=25%)' if recover/n >= 0.25 else '영상한계 지배 (재학습 무의미)'}")
    print("OOF_GATE_DONE")


if __name__ == "__main__":
    main()
