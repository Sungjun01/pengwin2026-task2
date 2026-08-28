"""Canonical official-aligned evaluator + gate for PENGWIN Task1.

Scores a predictions dir vs GT instance dir with official-aligned metrics and the
per-bone fragment-recovery diagnostic, and (optionally) reports the delta vs a
cached baseline (e.g. v18) so we can answer "does this candidate beat v18?".

Official alignment (verified — see progress_report_92/93, evaluator design wf):
  - LPS-orient pred and GT (mixed LPS/RAS in raw data; feedback_diag_orientation_trap)
  - remove connected components < 1000 mm3 from BOTH pred and GT before scoring
  - fragment matching = greedy highest-IoU one-to-one (PENGWIN 2024 spec; NOT hungarian)
  - instance match threshold IoU >= 0.5 (local default; sweep with --iou)

Metrics (reliable / approximate flagged):
  RELIABLE: instance_f1, instance_recall, instance_precision, merge_errors,
            split_errors, topology_consistency, fracture_dice (mean matched-frag Dice),
            score_dice (per-anatomy binary Dice), HD95_mm, ASSD_mm (bbox-efficient),
            per-bone fragment recovery ratio.
  APPROX/UNCONFIRMED official def: local_dice_cfs20 (surface-band Dice; official band
            construction unconfirmed) — reported but flag before using as ship gate.

Usage:
  python3 -m scripts.oof_11metric --pred_dir DIR --gt_dir DIR [--out scores.json]
                                  [--baseline v18_scores.json] [--no-hd95] [--iou 0.5]
"""
from __future__ import annotations
import argparse, json, glob, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

BONES = {"sacrum": (1, 50), "leftHip": (51, 100), "rightHip": (101, 150), "femur": (151, 200)}
MIN_CC_MM3 = 1000.0
LOCAL_BAND_MM = 20.0


def remove_small(lbl, vv, thr=MIN_CC_MM3):
    vals, cnts = np.unique(lbl, return_counts=True)
    drop = vals[(vals > 0) & (cnts * vv < thr)]
    if drop.size:
        keep = np.ones(int(lbl.max()) + 1, bool); keep[drop] = False
        lbl = lbl * keep[lbl]
    return lbl


def contingency(gt, pred):
    gt_ids = np.array(sorted(int(x) for x in np.unique(gt) if x > 0))
    pred_ids = np.array(sorted(int(x) for x in np.unique(pred) if x > 0))
    G, P = len(gt_ids), len(pred_ids)
    if G == 0 or P == 0:
        return gt_ids, pred_ids, np.zeros((G, P)), np.zeros(G), np.zeros(P)
    gl = np.zeros(int(gt.max()) + 1, np.int64); gl[gt_ids] = np.arange(1, G + 1)
    pl = np.zeros(int(pred.max()) + 1, np.int64); pl[pred_ids] = np.arange(1, P + 1)
    comb = (gl[gt] * (P + 1) + pl[pred]).ravel()
    cont = np.bincount(comb, minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1).astype(np.float64)
    return gt_ids, pred_ids, cont[1:, 1:], cont[1:, :].sum(1), cont[:, 1:].sum(0)


def _bbox(m, pad, shape):
    idx = np.where(m)
    if len(idx[0]) == 0:
        return None
    sl = tuple(slice(max(0, idx[d].min() - pad), min(shape[d], idx[d].max() + 1 + pad)) for d in range(3))
    return sl


def _hd95_assd(mg, mp, spacing):
    from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
    import torch
    sl = _bbox(mg | mp, 3, mg.shape)
    a = torch.from_numpy(mp[sl].astype(np.float32))[None, None]
    b = torch.from_numpy(mg[sl].astype(np.float32))[None, None]
    h = compute_hausdorff_distance(a, b, include_background=False, percentile=95, spacing=spacing)
    s = compute_average_surface_distance(a, b, include_background=False, symmetric=True, spacing=spacing)
    return float(h.item()), float(s.item())


def score_case(pf, gf, iou_thr=0.5, do_hd95=True):
    pim = sitk.DICOMOrient(sitk.ReadImage(pf), "LPS")
    gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
    pred = sitk.GetArrayFromImage(pim).astype(np.int32)
    gt = sitk.GetArrayFromImage(gim).astype(np.int32)
    if pred.shape != gt.shape:
        return {"error": f"shape {pred.shape} vs {gt.shape}"}
    sx, sy, sz = gim.GetSpacing(); vv = float(sx * sy * sz); spacing = (sz, sy, sx)
    pred = remove_small(pred, vv); gt = remove_small(gt, vv)
    gids, pids, inter, gsz, psz = contingency(gt, pred)
    G, P = len(gids), len(pids)
    union = gsz[:, None] + psz[None, :] - inter if G and P else np.zeros((G, P))
    iou = np.divide(inter, np.maximum(union, 1.0)) if G and P else np.zeros((G, P))
    dice = np.divide(2 * inter, np.maximum(gsz[:, None] + psz[None, :], 1.0)) if G and P else np.zeros((G, P))
    # greedy match @iou_thr
    used_g, used_p, matches = set(), set(), []
    for v, i, j in sorted(((iou[i, j], i, j) for i in range(G) for j in range(P) if iou[i, j] >= iou_thr), reverse=True):
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); matches.append((i, j))
    tp = len(matches); fn = G - tp; fp = P - tp
    rec = tp / (tp + fn) if tp + fn else (1.0 if P == 0 else 0.0)
    prec = tp / (tp + fp) if tp + fp else (1.0 if G == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    sub = inter > 0.2 * np.minimum(gsz[:, None], psz[None, :]) if G and P else np.zeros((G, P), bool)
    merge = int(sum(max(0, sub[:, j].sum() - 1) for j in range(P)))
    split = int(sum(max(0, sub[i, :].sum() - 1) for i in range(G)))
    denom = max(G, P, 1)
    topo = max(0.0, min(1.0, (denom - (merge + split)) / denom))
    mp = {i: j for i, j in matches}
    fdice = float(np.mean([dice[i, mp[i]] if i in mp else 0.0 for i in range(G)])) if G else 1.0
    # per-anatomy score dice (binary union per bone)
    sdices = []
    for lo, hi in BONES.values():
        bg = (gt >= lo) & (gt <= hi); bp = (pred >= lo) & (pred <= hi)
        if bg.any() or bp.any():
            sdices.append(2 * (bg & bp).sum() / max((bg.sum() + bp.sum()), 1))
    sdice = float(np.mean(sdices)) if sdices else 1.0
    # HD95/ASSD per matched frag (missed -> bounding sphere), bbox-efficient
    h95s, assds = [], []
    if do_hd95:
        for i in range(G):
            mg = gt == gids[i]
            if i in mp:
                h, s = _hd95_assd(mg, pred == pids[mp[i]], spacing)
            else:
                r = (3.0 * mg.sum() * vv / (4 * np.pi)) ** (1 / 3); h, s = 2 * r, r
            h95s.append(h); assds.append(s)
    bone = {}
    for bn, (lo, hi) in BONES.items():
        ng = int(((gids >= lo) & (gids <= hi)).sum()); npr = int(((pids >= lo) & (pids <= hi)).sum())
        if ng or npr:
            bone[bn] = {"gt": ng, "pred": npr}
    domain = "femur" if any(151 <= v <= 200 for v in gids) and not any(1 <= v <= 150 for v in gids) else "pelvic"
    return {"domain": domain, "n_gt": G, "n_pred": P, "instance_f1": f1, "instance_recall": rec,
            "instance_precision": prec, "merge_errors": float(merge), "split_errors": float(split),
            "topology_consistency": topo, "fracture_dice": fdice, "score_dice": sdice,
            "hd95_mm": float(np.mean(h95s)) if h95s else None, "assd_mm": float(np.mean(assds)) if assds else None,
            "bone": bone}


def aggregate(rows):
    keys = ["instance_f1", "instance_recall", "instance_precision", "merge_errors", "split_errors",
            "topology_consistency", "fracture_dice", "score_dice", "hd95_mm", "assd_mm"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    for bn in BONES:
        tg = sum(r["bone"][bn]["gt"] for r in rows if bn in r.get("bone", {}))
        tp = sum(r["bone"][bn]["pred"] for r in rows if bn in r.get("bone", {}))
        agg[f"frag_recovery_{bn}"] = (tp / tg) if tg else None
    agg["n"] = len(rows)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_prefix", default="PENGWIN_")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None, help="cached scores.json to compare against")
    ap.add_argument("--no-hd95", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    preds = sorted(glob.glob(f"{a.pred_dir}/*.mha")) + sorted(glob.glob(f"{a.pred_dir}/*.nii.gz"))
    rows = []
    for k, pf in enumerate(preds):
        num = Path(pf).name.split(".")[0].replace(a.gt_prefix, "")
        gf = None
        for cand in (f"{a.gt_dir}/{a.gt_prefix}{num}.nii.gz", f"{a.gt_dir}/{a.gt_prefix}{num}.mha", f"{a.gt_dir}/{num}.nii.gz"):
            if os.path.exists(cand):
                gf = cand; break
        if gf is None:
            continue
        r = score_case(pf, gf, iou_thr=a.iou, do_hd95=not a.no_hd95)
        if "error" not in r:
            r["case"] = num; rows.append(r)
        if (k + 1) % 30 == 0:
            print(f"  scored {k+1}/{len(preds)}", flush=True)
    agg = aggregate(rows)
    out = {"aggregate": agg, "per_case": rows}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n===== {a.pred_dir} (n={agg['n']}) =====")
    for k in ["instance_f1", "instance_recall", "instance_precision", "fracture_dice", "score_dice",
              "hd95_mm", "assd_mm", "merge_errors", "split_errors", "topology_consistency"]:
        print(f"  {k:22s} {agg[k]:.3f}" if agg[k] is not None else f"  {k:22s} (skipped)")
    print("  per-bone fragment recovery:")
    for bn in BONES:
        v = agg.get(f"frag_recovery_{bn}")
        if v is not None:
            print(f"    {bn:9s} {v:.3f}  {'UNDER-seg' if v < 0.95 else 'over-seg' if v > 1.05 else 'ok'}")
    if a.baseline and os.path.exists(a.baseline):
        base = json.load(open(a.baseline)).get("aggregate", {})
        print(f"\n===== delta vs baseline ({a.baseline}) =====")
        for k in ["instance_f1", "instance_recall", "instance_precision", "fracture_dice", "hd95_mm",
                  "frag_recovery_sacrum", "frag_recovery_leftHip", "frag_recovery_rightHip"]:
            if agg.get(k) is not None and base.get(k) is not None:
                d = agg[k] - base[k]
                print(f"  {k:24s} {base[k]:.3f} -> {agg[k]:.3f}  ({d:+.3f})")
    print("OOF_11METRIC_DONE")


if __name__ == "__main__":
    main()
