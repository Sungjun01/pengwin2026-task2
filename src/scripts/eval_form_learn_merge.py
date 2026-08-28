"""Merge-class P/R/F1 on labeled edges (§4A.8.1 primary gate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.form_learn_edge import EDGE_FEATURE_COLS


def merge_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def predict_logit(row: dict, model: dict) -> float:
    w = model["weights"]
    bias = float(model["bias"])
    means = model["feature_means"]
    stds = model["feature_stds"]
    feats = []
    for col in model.get("feature_cols", EDGE_FEATURE_COLS):
        v = float(row[col])
        mu = float(means.get(col, 0.0))
        sd = float(stds.get(col, 1.0)) or 1.0
        feats.append((v - mu) / sd)
    logit = bias + sum(wi * fi for wi, fi in zip(w, feats))
    return logit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=_ROOT / "progress" / "form_learn_merge.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.edges)
    labeled = df[(~df["abstain_flag"]) & df["y_merge"].notna()].copy()
    if labeled.empty:
        raise SystemExit("no labeled edges")

    y = labeled["y_merge"].astype(int).values
    if args.model.exists():
        model = json.loads(args.model.read_text())
        logits = labeled.apply(lambda r: predict_logit(r.to_dict(), model), axis=1).values
        scores = 1.0 / (1.0 + np.exp(-logits))
        mode = "learned"
    else:
        scores = labeled["e_rule_score"].values.astype(float)
        mode = "rule_only"

    m = merge_metrics(y, scores, args.threshold)
    print(f"[{mode}] labeled={len(labeled)} merge_pos={int(y.sum())}")
    print(f"  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  (thr={args.threshold})")

    if args.out:
        args.out.write_text(json.dumps({"mode": mode, **m}, indent=2))


if __name__ == "__main__":
    main()
