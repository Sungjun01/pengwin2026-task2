"""Case B 검증: Tversky(recall) + 전파-완성이 D016이 놓치는 *채점되는(>1cm³)* 조각을 회복하나?

decode-완성만으론 003서 슬리버(removed)만 생겨 null이었음. Tversky로 D016이 *실질적* border를
긋게 하면 전파-완성이 *큰(채점되는)* 조각으로 완성 → 회복 가능 가설.

D016 vs Tversky (둘 다 전파-완성 decode), 9 fold_3, 채점 사크럼 RQ + 채점 조각수 vs GT.
PASS = Tversky가 채점 조각수를 GT쪽으로 늘림(회복) + RQ↑ + 회귀 0.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.surface_completion_decode import propagation_completion_sacrum, rq_sac, lps
from evaluation.pengwin_eval import remove_small_components

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
D016V = "/home/schoaq/taehee/nnUNet_data/results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_3/validation"
TVV = "/home/schoaq/taehee/nnUNet_data/results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerGaussianSkeletonRecall__nnUNetPlans__3d_fullres/fold_3/validation"
CASES = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]
WCT, WDIST = 0.5, 1.0


def n_scored_sac(lbl, spz):
    s = np.where((lbl >= 1) & (lbl <= 50), lbl, 0).astype(np.uint16)
    s = remove_small_components(s, spz, 1000.0)
    return len([v for v in np.unique(s) if v > 0])


def main():
    print(f"=== Case B: Tversky+전파 vs D016+전파 (채점 사크럼 RQ + 채점 조각수, 9 OOF) ===")
    print(f"{'case':5s} {'GT조각':>6s} | {'D016 RQ':>8s} {'D016조각':>8s} | {'Tv RQ':>8s} {'Tv조각':>7s} | {'ΔRQ':>7s} {'Δ조각':>6s}")
    d_rqs, t_rqs = [], []; reg = []; recov = []
    for c in CASES:
        cid = f"PENGWIN_{c}"
        dp = Path(D016V) / f"{cid}.nii.gz"; tp = Path(TVV) / f"{cid}.nii.gz"
        gp = Path(RAW) / c / "label.mha"; ip = Path(RAW) / c / "image.mha"
        if not tp.exists():
            print(f"{c:5s}  Tversky 예측 없음 (학습/검증 미완)"); continue
        pim, dsem = lps(dp); sp = pim.GetSpacing(); spz = (sp[2], sp[1], sp[0])
        _, tsem = lps(tp); _, gt = lps(gp); _, ct = lps(ip); ct = ct.astype(np.float32)
        ng = n_scored_sac(gt, spz)
        d_inst = propagation_completion_sacrum(dsem, ct, k=2, w_ct=WCT, w_dist=WDIST)
        t_inst = propagation_completion_sacrum(tsem, ct, k=2, w_ct=WCT, w_dist=WDIST)
        d_rq = rq_sac(gt, d_inst, spz); t_rq = rq_sac(gt, t_inst, spz)
        d_n = n_scored_sac(d_inst, spz); t_n = n_scored_sac(t_inst, spz)
        d_rqs.append(d_rq); t_rqs.append(t_rq)
        drq = t_rq - d_rq; dn = t_n - d_n
        if drq < -0.02:
            reg.append(c)
        if t_rq > d_rq + 0.02:
            recov.append(c)
        flag = "  ↑회복" if drq > 0.02 else ("  ↓회귀" if drq < -0.02 else "")
        print(f"{c:5s} {ng:6d} | {d_rq:8.3f} {d_n:8d} | {t_rq:8.3f} {t_n:7d} | {drq:+7.3f} {dn:+6d}{flag}")
    if d_rqs:
        dm, tm = np.nanmean(d_rqs), np.nanmean(t_rqs)
        print(f"\n평균 채점 사크럼 RQ:  D016+전파={dm:.4f}  Tversky+전파={tm:.4f}  Δ={tm-dm:+.4f}")
        print(f"회복 케이스: {recov if recov else '없음'}  |  회귀: {reg if reg else '없음'}")
        go = (tm > dm + 0.005) and (len(reg) == 0)
        print(f"판정: {'GO — Tversky가 채점조각 회복, 회귀0 → fold_all 진행' if go else ('부분 — 회복 있으나 회귀 상존' if recov else 'NO — 회복 없음, Case B도 003엔 무력')}")
    print("CASEB_DONE")


if __name__ == "__main__":
    main()
