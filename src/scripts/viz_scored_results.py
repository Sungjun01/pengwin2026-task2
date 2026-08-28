"""기존 채점 결과 상세 시각화.

(A) OOF baseline 25케이스 ins_f1 막대 (0.80 평균선) — 어떤 케이스가 어려운가.
(B) Tversky 게이트(사크럼 RQ) D016 vs v1 vs v2, 9 fold_3 케이스 — 도박의 win/loss.
(C) 핵심 케이스 CT 단면: GT 조각(색칠) vs D016 예측 조각 — *왜* 그 점수가 나왔나
    (놓친 seam = GT는 2조각인데 D016이 한 덩어리로 병합).

용법: python -m scripts.viz_scored_results
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"

# 기록된 채점 (b4afqq0cj OOF baseline 25케이스 ins_f1, legacy)
OOF_BASE = {"006":0.60,"063":0.86,"084":0.62,"064":0.82,"198":0.82,"075":0.82,"115":0.93,
            "179":0.83,"184":0.86,"087":0.60,"098":0.80,"108":0.91,"120":0.88,"167":0.91,
            "171":0.67,"182":0.80,"196":0.77,"097":0.71,"100":0.93,"164":0.80,"173":0.67,
            "040":0.82,"050":0.93,"058":0.70,"067":0.92}
# Tversky 게이트(사크럼 RQ) 9 fold_3 케이스: (D016, v1=β0.7/λ0.3, v2=λ0.2/wdice0.3)
TV_GATE = {"006":(0.429,0.667,0.429),"063":(0.800,0.667,0.727),"084":(0.500,0.500,0.444),
           "198":(0.571,0.571,0.571),"075":(0.727,0.727,0.727),"108":(0.857,0.667,0.857),
           "182":(0.667,0.667,0.800),"097":(0.400,0.667,0.667),"058":(0.857,0.750,0.750)}
KEY_CASES = ["097", "063", "006", "058"]  # 분리win, 회귀, 어려운, 좋은


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac_relabel(lbl):
    """사크럼(1-50) instance 를 1..K 로 relabel (시각화용)."""
    out = np.zeros_like(lbl, dtype=np.int32)
    for i, v in enumerate([v for v in np.unique(lbl) if 1 <= v <= 50], 1):
        out[lbl == v] = i
    return out, int(out.max())


def best_slice(sac_gt):
    """사크럼 라벨이 가장 많이 보이는 axial z."""
    best, bz = -1, sac_gt.shape[0] // 2
    for z in range(sac_gt.shape[0]):
        n = len(np.unique(sac_gt[z])) - 1
        if n > best:
            best, bz = n, z
    return bz


def main():
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.4])

    # (A) OOF baseline 막대
    axA = fig.add_subplot(gs[0, :])
    items = sorted(OOF_BASE.items(), key=lambda kv: kv[1])
    cs = ["#d62728" if v < 0.75 else ("#ff7f0e" if v < 0.85 else "#2ca02c") for _, v in items]
    axA.bar([k for k, _ in items], [v for _, v in items], color=cs)
    axA.axhline(0.80, ls="--", c="k", lw=1); axA.text(0.5, 0.81, "평균 0.80", fontsize=9)
    axA.set_title("(A) OOF(held-out) baseline ins_f1 — 25 사크럼-골절 케이스 (빨강<0.75 어려움)", fontsize=12)
    axA.set_ylabel("ins_f1"); axA.set_ylim(0, 1); axA.tick_params(axis='x', labelsize=8)

    # (B) Tversky 게이트 grouped
    axB = fig.add_subplot(gs[1, :])
    ks = list(TV_GATE.keys()); x = np.arange(len(ks)); w = 0.27
    d016 = [TV_GATE[k][0] for k in ks]; v1 = [TV_GATE[k][1] for k in ks]; v2 = [TV_GATE[k][2] for k in ks]
    axB.bar(x - w, d016, w, label="D016 baseline", color="#7f7f7f")
    axB.bar(x, v1, w, label="Tversky v1 (β0.7/λ0.3)", color="#1f77b4")
    axB.bar(x + w, v2, w, label="Tversky v2 (λ0.2/wdice0.3)", color="#ff7f0e")
    for i, k in enumerate(ks):
        if v1[i] > d016[i] + 0.02 or v2[i] > d016[i] + 0.02:
            axB.text(x[i], max(v1[i], v2[i]) + 0.02, "↑win", ha="center", color="green", fontsize=8)
        if v1[i] < d016[i] - 0.02 or v2[i] < d016[i] - 0.02:
            axB.text(x[i], d016[i] + 0.02, "↓회귀", ha="center", color="red", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels(ks); axB.set_ylabel("사크럼 RQ"); axB.set_ylim(0, 1)
    axB.set_title("(B) Tversky 게이트 — 사크럼 RQ (097 분리win 굳건 / 063 회귀 굳건 / 평균 +0.018·회귀 상존 → NO-GO)", fontsize=12)
    axB.legend(fontsize=9)

    # (C) 핵심 케이스 CT 오버레이: GT vs D016
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    for ci, c in enumerate(KEY_CASES):
        _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
        _, gt = lps(Path(RAW) / c / "label.mha")
        gimg, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz")
        spx = gimg.GetSpacing(); vv = float(np.prod(spx)); spz = (spx[2], spx[1], spx[0])
        pred = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        gsac, ng = sac_relabel(gt); psac, npd = sac_relabel(pred)
        z = best_slice(gsac)
        for col, (lab, arr, n) in enumerate([("GT", gsac, ng), ("D016 예측", psac, npd)]):
            ax = fig.add_subplot(gs[2, ci] if col == 0 else None) if False else fig.add_axes([
                0.02 + ci * 0.245 + col * 0.12, 0.02, 0.11, 0.26])
            ax.imshow(ct[z], cmap="gray", vmin=-200, vmax=1000)
            m = np.ma.masked_where(arr[z] == 0, arr[z])
            ax.imshow(m, cmap=cmap, alpha=0.55, vmin=1, vmax=20)
            d016rq = TV_GATE.get(c, (None,))[0]
            ax.set_title(f"{c} {lab}\nGT {ng}조각 vs 예측 {npd}조각" + (f"\nRQ={d016rq:.2f}" if col == 0 and d016rq else ""), fontsize=8)
            ax.axis("off")
    fig.text(0.5, 0.30, "(C) 핵심 케이스 — CT + 사크럼 조각(색=조각번호): GT vs D016 예측 "
             "[097=D016이 1조각으로 병합(Tversky가 분리), 063=둘다 잘 분리]", ha="center", fontsize=11)

    out = "figs/scored_results_detail.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    print("VIZ_SCORED_DONE")


if __name__ == "__main__":
    main()
