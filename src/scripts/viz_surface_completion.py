"""면-완성 decode 시각화: CT + [GT | legacy | surface-completion(k)] 사크럼 조각.

면-완성이 *실제로* 끊긴 seam 을 이어 조각을 분리하는지 눈으로 검증.
용법: python -m scripts.viz_surface_completion --k 2 --cases 097 063 006
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.family"] = "DejaVu Sans"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.surface_completion_decode import surface_completion_sacrum, rq_sac, lps

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"


def relbl(lbl):
    out = np.zeros_like(lbl, dtype=np.int32)
    for i, v in enumerate([v for v in np.unique(lbl) if 1 <= v <= 50], 1):
        out[lbl == v] = i
    return out, int(out.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cases", nargs="+", default=["097", "063", "006"])
    args = ap.parse_args()
    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    rows = args.cases
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for ri, c in enumerate(rows):
        pim, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz"); sp = pim.GetSpacing()
        vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
        _, gt = lps(Path(RAW) / c / "label.mha")
        leg = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        sc = surface_completion_sacrum(sem, args.k)
        g_rl, ng = relbl(gt); l_rl, nl = relbl(leg); s_rl, ns = relbl(sc)
        leg_rq = rq_sac(gt, leg, spz); sc_rq = rq_sac(gt, sc, spz)
        z = max(range(g_rl.shape[0]), key=lambda zz: len(np.unique(g_rl[zz])))
        panels = [("GT", g_rl, ng, None), (f"legacy", l_rl, nl, leg_rq),
                  (f"surface-completion k={args.k}", s_rl, ns, sc_rq)]
        for ci, (lab, arr, n, rq) in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(ct[z], cmap="gray", vmin=-200, vmax=1000)
            ax.imshow(np.ma.masked_where(arr[z] == 0, arr[z]), cmap=cmap, alpha=0.55, vmin=1, vmax=20)
            t = f"case {c} — {lab}\n{n} sacrum frags"
            if rq is not None:
                t += f" | RQ={rq:.2f}"
            ax.set_title(t, fontsize=10); ax.axis("off")
    plt.suptitle(f"Surface-completion decode (k={args.k}): GT vs legacy vs completed. "
                 "More fragments + higher RQ = recovered seam (retrain-free).", fontsize=12)
    plt.tight_layout()
    out = f"figs/surface_completion_k{args.k}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)
    print("VIZ_SURFCOMP_DONE")


if __name__ == "__main__":
    main()
