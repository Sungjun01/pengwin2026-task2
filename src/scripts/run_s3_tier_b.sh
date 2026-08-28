#!/usr/bin/env bash
# Tier B Form-Learn pipeline (§4A.6 / §4A.8)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source scripts/nnunet_env.sh 2>/dev/null || true

PROBS="${PROBS:-$REPO/predictions/d015_probs_heldout}"
EDGES_TRAIN="${EDGES_TRAIN:-$REPO/progress/form_learn_edges_train.parquet}"
EDGES_HO="${EDGES_HO:-$REPO/progress/form_learn_edges_heldout.parquet}"
MODEL="${MODEL:-$REPO/progress/form_learn_merge.json}"

echo "=== 1) Export held-out D015 probabilities (optional, skip if exists) ==="
mkdir -p "$PROBS"
python3 scripts/export_d015_probs.py --split heldout --out_dir "$PROBS" || true

echo "=== 2) Build edge datasets ==="
python3 scripts/build_form_learn_edge_dataset.py \
  --pred_dir predictions/heldout_d015_ep999_foldall \
  --probs_dir "$PROBS" \
  --split heldout --out "$EDGES_HO"

python3 scripts/build_form_learn_edge_dataset.py \
  --semantic_dir "$nnUNet_raw/Dataset015_PENGWIN_BorderCore_OrientFixed/labelsTr" \
  --probs_dir "$PROBS" \
  --split train --out "$EDGES_TRAIN"

echo "=== 3) Train merge head ==="
python3 scripts/train_form_learn_merge.py --edges "$EDGES_TRAIN" "$EDGES_HO" --out "$MODEL"

echo "=== 4) Merge F1 gate ==="
python3 scripts/eval_form_learn_merge.py --edges "$EDGES_HO" --model "$MODEL" \
  --out progress/form_learn_merge_metrics_heldout.json

echo "=== 5) Postproc modes on held-out 6 ==="
python3 scripts/eval_form_learn_heldout.py

echo "=== 6) Co-flip test ==="
python3 -m pytest scripts/tests/test_d015_pivot_coflip.py -q

echo "[done] Tier B pipeline"
