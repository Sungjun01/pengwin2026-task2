"""Evaluate a trained SAFT checkpoint on the held-out pelvic hard cases.

Runs inference (affinity decode, and boundary decode for comparison) on each
held-out case, then scores instance F1 + foreground Dice against the raw
PENGWIN fragment GT (predictions/gt_instance), and compares to the pre-fix
baseline (instance_f1 ~0.005).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.infer_saft_case import infer_saft_case_from_paths
from scripts.saft_eval_metrics import evaluate_instance_quality
from scripts.voxel_mamba_io import load_lps_volume

REPO = Path(__file__).resolve().parents[1]
CT_DIR = Path("/home/schoaq/taehee/nnUNet_data/raw/Dataset001_PENGWIN_Anatomical/imagesTr")
GT_DIR = REPO / "predictions" / "gt_instance"
CKPT = REPO / "logs" / "saft_pelvic_fixed_f48_p96" / "best.pt"
OUT_DIR = REPO / "logs" / "saft_pelvic_fixed_f48_p96" / "heldout_eval"
CASES = ["004", "084", "087"]
BASELINE = {"instance_f1": 0.005128, "Dice_A_mean": 0.004057, "Dice_F_binary": 0.239777, "n_pred": 97.0}


def _binary_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt > 0
    p = pred > 0
    denom = int(g.sum()) + int(p.sum())
    return 2.0 * int((g & p).sum()) / denom if denom else 1.0


def _n_ids(lbl: np.ndarray) -> int:
    return len(set(int(v) for v in np.unique(lbl)) - {0})


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for decode in ("affinity", "boundary"):
        per_case = {}
        for case in CASES:
            ct = CT_DIR / f"PENGWIN_{case}_0000.nii.gz"
            gt_path = GT_DIR / f"PENGWIN_{case}.nii.gz"
            out_path = OUT_DIR / f"{case}_{decode}_pred.mha"
            infer_saft_case_from_paths(CKPT, ct, out_path, min_voxels=20, device="cuda", decode=decode)
            gt = load_lps_volume(gt_path, dtype=np.int32).array
            pred = load_lps_volume(out_path, dtype=np.int32).array
            iq = evaluate_instance_quality(gt, pred, iou_threshold=0.5)
            per_case[case] = {
                **iq,
                "Dice_F_binary": _binary_dice(gt, pred),
                "n_gt": _n_ids(gt),
                "n_pred": _n_ids(pred),
            }
            print(json.dumps({"decode": decode, "case": case, **per_case[case]}, sort_keys=True), flush=True)
        agg_keys = ["instance_f1", "instance_recall", "instance_precision", "Dice_F_binary", "n_pred", "n_gt"]
        results[decode] = {
            "per_case": per_case,
            "mean": {k: float(np.mean([per_case[c][k] for c in CASES])) for k in agg_keys},
        }

    summary = {
        "baseline_prefix": BASELINE,
        "affinity_mean": results["affinity"]["mean"],
        "boundary_mean": results["boundary"]["mean"],
        "results": results,
    }
    (OUT_DIR / "heldout_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "eval_done",
                      "baseline_instance_f1": BASELINE["instance_f1"],
                      "fixed_affinity_instance_f1": results["affinity"]["mean"]["instance_f1"],
                      "fixed_boundary_instance_f1": results["boundary"]["mean"]["instance_f1"]},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
