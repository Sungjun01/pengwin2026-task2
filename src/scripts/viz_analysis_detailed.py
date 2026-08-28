"""상세·다양 분석 시각화 (영어 라벨 — 한글폰트 부재로 깨짐 방지).

fig1 analysis_scores.png : (A) OOF baseline 25 (B) Tversky gate D016/v1/v2 (C) decode-mode sweep
fig2 analysis_wall.png   : (D) seam HU contrast found vs missed (E) missed-seam atlas×HU breakdown (F) 1st vs v18 gap
fig3 analysis_cases.png  : 097(collapse→split) & 063(regression) 큰 CT, axial+coronal, GT vs D016 조각

용법: python -m scripts.viz_analysis_detailed
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"

OOF_BASE = {"006":0.60,"063":0.86,"084":0.62,"064":0.82,"198":0.82,"075":0.82,"115":0.93,
            "179":0.83,"184":0.86,"087":0.60,"098":0.80,"108":0.91,"120":0.88,"167":0.91,
            "171":0.67,"182":0.80,"196":0.77,"097":0.71,"100":0.93,"164":0.80,"173":0.67,
            "040":0.82,"050":0.93,"058":0.70,"067":0.92}
TV_GATE = {"006":(0.429,0.667,0.429),"063":(0.800,0.667,0.727),"084":(0.500,0.500,0.444),
           "198":(0.571,0.571,0.571),"075":(0.727,0.727,0.727),"108":(0.857,0.667,0.857),
           "182":(0.667,0.667,0.800),"097":(0.400,0.667,0.667),"058":(0.857,0.750,0.750)}
# decode-mode 스윕 (clean OOF, b4afqq0cj)
DECODE = [("legacy/500",0.7998),("keep_all/500",0.8038),("hybrid/500",0.8038),
          ("keep_all/250",0.8017),("legacy/250",0.7977),("keep_all/1000",0.7976),
          ("legacy/1000",0.7936),("form_intact",0.5322)]
# atlas↔error (n=114): atlas-HIGH 44(vis24/invis20), atlas-LOW 70(vis28/invis42)
ATLAS_ERR = {"HIGH":(24,20),"LOW":(28,42)}
# 1st vs v18 per-case (공식 JSON)
GAP = {"002":(1.0,1.0),"003":(1.0,0.952),"004":(1.0,1.0),"005":(1.0,1.0),"001":(0.833,0.75)}


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac_relabel(lbl):
    out = np.zeros_like(lbl, dtype=np.int32)
    for i, v in enumerate([v for v in np.unique(lbl) if 1 <= v <= 50], 1):
        out[lbl == v] = i
    return out, int(out.max())


def fig_scores():
    fig, ax = plt.subplots(3, 1, figsize=(15, 12))
    items = sorted(OOF_BASE.items(), key=lambda kv: kv[1])
    cs = ["#d62728" if v < 0.75 else ("#ff7f0e" if v < 0.85 else "#2ca02c") for _, v in items]
    ax[0].bar([k for k, _ in items], [v for _, v in items], color=cs)
    ax[0].axhline(0.80, ls="--", c="k"); ax[0].text(0.3, 0.815, "mean 0.80")
    ax[0].set_title("(A) Clean OOF (held-out) instance-F1 — 25 sacrum-fracture cases  [red<0.75 hard, green>=0.85 good]")
    ax[0].set_ylabel("instance F1"); ax[0].set_ylim(0, 1)

    ks = list(TV_GATE.keys()); x = np.arange(len(ks)); w = 0.27
    ax[1].bar(x - w, [TV_GATE[k][0] for k in ks], w, label="D016 baseline", color="#7f7f7f")
    ax[1].bar(x, [TV_GATE[k][1] for k in ks], w, label="Tversky v1 (b0.7/l0.3)", color="#1f77b4")
    ax[1].bar(x + w, [TV_GATE[k][2] for k in ks], w, label="Tversky v2 (l0.2/wdice0.3)", color="#ff7f0e")
    for i, k in enumerate(ks):
        d = TV_GATE[k]
        if max(d[1], d[2]) > d[0] + 0.02: ax[1].text(x[i], max(d[1], d[2]) + 0.02, "win", ha="center", color="green", fontsize=8)
        if min(d[1], d[2]) < d[0] - 0.02: ax[1].text(x[i], d[0] + 0.03, "regress", ha="center", color="red", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ks); ax[1].set_ylim(0, 1); ax[1].set_ylabel("sacrum RQ")
    ax[1].set_title("(B) Tversky gate (sacrum RQ): 097 split-win robust / 063 regress robust / mean +0.018 but regressions persist -> NO-GO")
    ax[1].legend()

    lbls = [d[0] for d in DECODE]; vals = [d[1] for d in DECODE]
    cb = ["#2ca02c" if "legacy/500" in l else "#7f7f7f" for l in lbls]
    ax[2].bar(lbls, vals, color=cb); ax[2].axhline(0.7998, ls="--", c="green")
    ax[2].set_ylim(0.4, 0.85); ax[2].set_ylabel("mean OOF ins-F1")
    ax[2].set_title("(C) Decode-mode sweep (no retrain): legacy=optimal (+0.004 best) -> decode-fix is DEAD. form_intact collapses.")
    ax[2].tick_params(axis='x', labelsize=8)
    plt.tight_layout(); plt.savefig("figs/analysis_scores.png", dpi=110, bbox_inches="tight"); plt.close()
    print("saved figs/analysis_scores.png")


def fig_wall():
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    # (D) found vs missed seam HU contrast (개략, report 76 + OOF bins)
    ax[0].axvline(96, c="#2ca02c", lw=2, label="FOUND median 96 HU (n=115)")
    ax[0].axvline(22, c="#d62728", lw=2, ls="--", label="MISSED median 22 HU")
    ax[0].axvspan(-50, 40, alpha=0.12, color="red"); ax[0].text(-40, 0.8, "imaging limit\n(<40HU, ~invisible)", color="#d62728", fontsize=9)
    ax[0].axvspan(40, 200, alpha=0.08, color="green"); ax[0].text(110, 0.8, "recoverable\n(>=40HU)", color="#2ca02c", fontsize=9)
    ax[0].set_xlim(-50, 200); ax[0].set_ylim(0, 1); ax[0].set_yticks([])
    ax[0].set_xlabel("seam bone-vs-gap HU contrast"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("(D) The 22HU wall: missed seams are near-invisible in CT")

    # (E) missed-seam atlas x HU breakdown (n=114)
    cats = ["atlas-HIGH\n(predictable)", "atlas-LOW\n(unpredictable)"]
    vis = [ATLAS_ERR["HIGH"][0], ATLAS_ERR["LOW"][0]]; invis = [ATLAS_ERR["HIGH"][1], ATLAS_ERR["LOW"][1]]
    ax[1].bar(cats, vis, color="#2ca02c", label="visible (>=40HU)")
    ax[1].bar(cats, invis, bottom=vis, color="#d62728", label="invisible (<40HU)")
    ax[1].text(0, vis[0]/2, f"{vis[0]}\n(TARGET\n21%)", ha="center", color="white", fontsize=9, fontweight="bold")
    ax[1].set_ylabel("# missed seams (n=114)"); ax[1].legend(fontsize=8)
    ax[1].set_title("(E) Missed-seam breakdown: only atlas-HIGH&visible (21%) is recoverable")

    # (F) 1st vs v18 gap
    ks = list(GAP.keys()); x = np.arange(len(ks)); w = 0.35
    ax[2].bar(x - w/2, [GAP[k][0] for k in ks], w, label="1st place (0.967)", color="#1f77b4")
    ax[2].bar(x + w/2, [GAP[k][1] for k in ks], w, label="our v18 (0.940)", color="#ff7f0e")
    ax[2].text(1, 0.80, "003: miss 1\nfragment\n(recall)", ha="center", fontsize=8)
    ax[2].text(4, 0.55, "001: boundary\nIoU quality", ha="center", fontsize=8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(ks); ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("ins F1")
    ax[2].set_title("(F) Gap to 1st: only 003 (recall) + 001 (boundary). 002/004/005 maxed.")
    ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("figs/analysis_wall.png", dpi=110, bbox_inches="tight"); plt.close()
    print("saved figs/analysis_wall.png")


def fig_cases():
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    cases = [("097", "D016 collapses sacrum -> Tversky splits (win)"),
             ("063", "both split well -> Tversky over-pushes (regress)")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ri, (c, note) in enumerate(cases):
        _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
        _, gt = lps(Path(RAW) / c / "label.mha")
        gimg, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz")
        spx = gimg.GetSpacing(); vv = float(np.prod(spx)); spz = (spx[2], spx[1], spx[0])
        pred = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        gsac, ng = sac_relabel(gt); psac, npd = sac_relabel(pred)
        # axial best slice
        za = max(range(gsac.shape[0]), key=lambda z: len(np.unique(gsac[z])))
        # coronal best slice (axis1)
        yc = max(range(gsac.shape[1]), key=lambda y: len(np.unique(gsac[:, y])))
        panels = [("axial GT", ct[za], gsac[za], ng), ("axial D016", ct[za], psac[za], npd),
                  ("coronal GT", ct[:, yc], gsac[:, yc], ng), ("coronal D016", ct[:, yc], psac[:, yc], npd)]
        for ci, (lab, bg, ov, n) in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(bg, cmap="gray", vmin=-200, vmax=1000, aspect="auto")
            ax.imshow(np.ma.masked_where(ov == 0, ov), cmap=cmap, alpha=0.55, vmin=1, vmax=20, aspect="auto")
            ax.set_title(f"case {c} — {lab}\n{n} sacrum fragments", fontsize=10)
            ax.axis("off")
        axes[ri, 0].text(-0.05, 0.5, f"case {c}\n{note}", transform=axes[ri, 0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
    plt.suptitle("Case detail: GT sacrum fragments vs D016 prediction (color = fragment id). "
                 "Fewer pred fragments than GT = missed seam (merge).", fontsize=13)
    plt.tight_layout(); plt.savefig("figs/analysis_cases.png", dpi=110, bbox_inches="tight"); plt.close()
    print("saved figs/analysis_cases.png")


def main():
    fig_scores(); fig_wall(); fig_cases()
    print("VIZ_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
