"""Train Tier B merge logistic head on edge parquet (§4A.8.2)."""
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

DEFAULT_OUT = _ROOT / "progress" / "form_learn_merge.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--edges",
        type=Path,
        nargs="+",
        default=[
            _ROOT / "progress" / "form_learn_edges_train_gtbc.parquet",
            _ROOT / "progress" / "form_learn_edges_heldout.parquet",
        ],
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frames = []
    for p in args.edges:
        if p.exists() and p.stat().st_size > 0:
            frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("no edge parquet files found")

    df = pd.concat(frames, ignore_index=True)
    train_df = df[(~df["abstain_flag"]) & df["y_merge"].notna()].copy()
    if len(train_df) < 10:
        raise SystemExit(f"too few labeled edges: {len(train_df)}")

    cols = [c for c in EDGE_FEATURE_COLS if c in train_df.columns]
    X = train_df[cols].astype(np.float64).values
    y = train_df["y_merge"].astype(int).values
    print(f"[train] edges={len(train_df)}  merge_pos={int(y.sum())}  merge_neg={int((1-y).sum())}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=args.C, class_weight="balanced", max_iter=500)),
        ]
    )
    if len(np.unique(y)) > 1 and len(train_df) >= 20:
        proba = cross_val_predict(pipe, X, y, cv=min(5, len(train_df)), method="predict_proba")[:, 1]
        try:
            auc = roc_auc_score(y, proba)
            pred = (proba >= 0.5).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            print(f"[cv] merge AUPRC proxy auc={auc:.3f}  f1={f1:.3f}")
        except ValueError:
            pass

    pipe.fit(X, y)
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    cols = [c for c in EDGE_FEATURE_COLS if c in train_df.columns]
    payload = {
        "feature_cols": cols,
        "feature_means": {c: float(m) for c, m in zip(EDGE_FEATURE_COLS, scaler.mean_)},
        "feature_stds": {c: float(s) for c, s in zip(EDGE_FEATURE_COLS, scaler.scale_)},
        "weights": [float(w) for w in clf.coef_[0]],
        "bias": float(clf.intercept_[0]),
        "theta_merge_default": 0.35,
        "n_train": int(len(train_df)),
        "n_merge_pos": int(y.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
