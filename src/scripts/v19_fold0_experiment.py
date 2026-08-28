"""Fast V19-vs-v18(baseline) decode comparison on Aseg fold_0 held-out (34 cases).

Decode the SAME semantic predictions two ways:
  baseline = v18 canonical decode (26-conn core-CC + watershed)  [v19_decode.baseline_config]
  v19      = region-aware separation decode                       [v19_decode.v19_config]
Score instance recall / precision / merge / split with the panoptic IoU>=0.5
matcher (same as scripts/decode_variants.py) so this is a clean merge-axis read
BEFORE the slower surface-metric harness gate.

Verdict (merge axis): WIN if recall up AND merge down WITHOUT split exploding.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee")
sys.path.insert(0, str(REPO / "scripts"))
import v19_decode as V19  # noqa: E402

GT_DIR = REPO / "predictions" / "gt_instance"
ASEG_VAL = glob.glob(
    "/home/schoaq/taehee/nnUNet_data/results/Dataset016*/"
    "nnUNetTrainerBorderWeightedAseg_w40__*/fold_0/validation"
)[0]
MIN_CC_MM3 = V19.DEFAULT_MIN_CC_MM3


def remove_small(lbl, vv):
    vals, cnts = np.unique(lbl, return_counts=True)
    drop = vals[(vals > 0) & (cnts * vv < MIN_CC_MM3)]
    if drop.size:
        keep = np.ones(int(lbl.max()) + 1, bool)
        keep[drop] = False
        lbl = lbl * keep[lbl]
    return lbl


def score(gt, pred, vv):
    gt = remove_small(gt.astype(np.int32), vv)
    pred = remove_small(pred.astype(np.int32), vv)
    gids = np.array(sorted(int(x) for x in np.unique(gt) if x > 0))
    pids = np.array(sorted(int(x) for x in np.unique(pred) if x > 0))
    G, P = len(gids), len(pids)
    if G == 0:
        return 1.0, 1.0, 1.0, 0, 0
    if P == 0:
        return 0.0, 0.0, 0.0, 0, 0
    gl = np.zeros(int(gt.max()) + 1, np.int64); gl[gids] = np.arange(1, G + 1)
    pl = np.zeros(int(pred.max()) + 1, np.int64); pl[pids] = np.arange(1, P + 1)
    cont = np.bincount((gl[gt] * (P + 1) + pl[pred]).ravel(),
                       minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1).astype(np.float64)
    inter = cont[1:, 1:]; gsz = cont[1:, :].sum(1); psz = cont[:, 1:].sum(0)
    iou = inter / np.maximum(gsz[:, None] + psz[None, :] - inter, 1.0)
    ug, up, tp = set(), set(), 0
    for v, i, j in sorted(((iou[i, j], i, j) for i in range(G) for j in range(P)
                           if iou[i, j] >= 0.5), reverse=True):
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); tp += 1
    rec = tp / G; prec = tp / P; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    sub = inter > 0.2 * np.minimum(gsz[:, None], psz[None, :])
    merge = sum(max(0, sub[:, j].sum() - 1) for j in range(P))
    split = sum(max(0, sub[i, :].sum() - 1) for i in range(G))
    return f1, rec, prec, merge, split


def main():
    sems = sorted(glob.glob(f"{ASEG_VAL}/*.nii.gz"))
    arms = {
        "baseline": V19.baseline_config(),
        "v19_conn": V19.v19_config(),
        "v19_border": V19.v19_border_aware_config(border_close_iters=1),
        "v19_border2": V19.v19_border_aware_config(border_close_iters=2),
    }
    agg = {k: [] for k in arms}
    n = 0
    for sf in sems:
        cid = os.path.basename(sf).split(".")[0]
        gf = GT_DIR / f"{cid}.nii.gz"
        if not gf.exists():
            continue
        sim = sitk.DICOMOrient(sitk.ReadImage(sf), "LPS")
        gim = sitk.DICOMOrient(sitk.ReadImage(str(gf)), "LPS")
        sem = sitk.GetArrayFromImage(sim).astype(np.int32)
        gt = sitk.GetArrayFromImage(gim).astype(np.int32)
        if sem.shape != gt.shape:
            continue
        sx, sy, sz = gim.GetSpacing(); vv = float(sx * sy * sz)
        for name, cfg in arms.items():
            pred = V19.decode_volume(sem, (sx, sy, sz), cfg)
            agg[name].append(score(gt, pred, vv))
        n += 1
        print(f"  scored {cid} ({n})", flush=True)

    print(f"\n=== V19 vs baseline (Aseg fold_0 held-out, n={n}) ===", flush=True)
    print(f"{'arm':>9} {'F1':>6} {'recall':>7} {'prec':>6} {'merge':>6} {'split':>6}")
    for name in arms:
        A = np.array(agg[name])
        print(f"{name:>9} {A[:,0].mean():>6.3f} {A[:,1].mean():>7.3f} {A[:,2].mean():>6.3f} "
              f"{A[:,3].mean():>6.2f} {A[:,4].mean():>6.2f}")
    b = np.array(agg["baseline"])
    for name in arms:
        if name == "baseline":
            continue
        v = np.array(agg[name])
        df = v[:,0].mean() - b[:,0].mean()
        dr = v[:,1].mean() - b[:,1].mean(); dm = v[:,3].mean() - b[:,3].mean(); ds = v[:,4].mean() - b[:,4].mean()
        verdict = "WIN" if (df > 0.005 and dm < 0 and ds < 0.5) else ("split-cost" if ds >= 0.5 else "~flat")
        print(f"  {name:>12}: ΔF1={df:+.3f} Δrecall={dr:+.3f}  Δmerge={dm:+.2f}  Δsplit={ds:+.2f}  -> {verdict}")
    print("V19_FOLD0_DONE", flush=True)


if __name__ == "__main__":
    main()
