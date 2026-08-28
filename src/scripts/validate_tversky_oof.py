"""Tversky fine-tune의 OOF(held-out) 검증 — 깨끗, 도박의 GO/NO-GO 게이트.

fold_3 held-out 사크럼-골절 9케이스에서 Tversky fine-tuned 모델 vs D016 baseline 의
사크럼 instance RQ(legacy decode) 비교. 둘 다 fold_3 의 validation 예측(=각 케이스를
학습 안 함 = OOF). nnUNet 이 학습 끝에 자동 생성한 validation/*.nii.gz 를 그대로 사용.

PASS(도박 진행) 조건: (a) 평균 사크럼 RQ 가 D016 보다 ↑, (b) 어떤 케이스도 회귀(−0.02 초과) 0.
FAIL → fold_all 학습 중단, v18 유지.

용법: python -m scripts.validate_tversky_oof
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.gmm_sliver_merge import rq_sac, lps

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
RES = "/home/schoaq/taehee/nnUNet_data/results/Dataset016_PENGWIN_BorderCore_MedialOrient"
TVERSKY = f"{RES}/nnUNetTrainerGaussianSkeletonRecall__nnUNetPlans__3d_fullres/fold_3/validation"
D016 = f"{RES}/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_3/validation"
CASES = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]  # fold_3 held-out sacrum-frac


def n_sac(lbl):
    return len([v for v in np.unique(lbl) if 1 <= v <= 50])


def decode_score(pred_path, gt):
    pim, sem = lps(pred_path)
    sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
    inst = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
    return rq_sac(gt, inst, spz), n_sac(inst)


def main():
    print("=== Tversky OOF 검증 (fold_3 held-out 9 사크럼-골절, legacy decode) ===")
    print(f"{'case':6s} {'GT조각':>6s} {'D016 RQ':>8s} {'Tv RQ':>8s} {'Δ':>7s} {'D016조각':>7s} {'Tv조각':>7s}")
    d_rqs, t_rqs = [], []
    regress = []
    for c in CASES:
        cid = f"PENGWIN_{c}"
        dp = Path(D016) / f"{cid}.nii.gz"; tp = Path(TVERSKY) / f"{cid}.nii.gz"
        gp = Path(RAW) / c / "label.mha"
        if not tp.exists():
            print(f"{c:6s}  Tversky 예측 없음 (학습/검증 미완?) -> {tp}"); continue
        if not (dp.exists() and gp.exists()):
            print(f"{c:6s}  baseline/GT 없음"); continue
        _, gt = lps(gp); ng = n_sac(gt)
        d_rq, d_n = decode_score(dp, gt)
        t_rq, t_n = decode_score(tp, gt)
        d_rqs.append(d_rq); t_rqs.append(t_rq)
        delta = t_rq - d_rq
        if delta < -0.02:
            regress.append((c, delta))
        flag = "  <<recall↑" if delta > 0.02 else ("  회귀!" if delta < -0.02 else "")
        print(f"{c:6s} {ng:6d} {d_rq:8.3f} {t_rq:8.3f} {delta:+7.3f} {d_n:7d} {t_n:7d}{flag}")
    if d_rqs:
        dm, tm = np.nanmean(d_rqs), np.nanmean(t_rqs)
        print(f"\n평균 사크럼 RQ:  D016={dm:.4f}  Tversky={tm:.4f}  Δ={tm-dm:+.4f}")
        print(f"회귀 케이스(Δ<-0.02): {[c for c,_ in regress] if regress else '없음'}")
        go = (tm > dm + 0.005) and (len(regress) == 0)
        print(f"\n판정: {'GO — fold_all 학습 진행 (recall↑, 비회귀)' if go else 'NO-GO — v18 유지 (개선없음 or 회귀)'}")
    print("TVERSKY_OOF_DONE")


if __name__ == "__main__":
    main()
