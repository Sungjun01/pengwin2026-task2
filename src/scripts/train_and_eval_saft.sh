#!/bin/bash
# Full SAFT pipeline: retrain on pelvic cases (fixed code), then evaluate held-out hard cases.
set -e
REPO=/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO"
PY=.venv_voxel_mamba/bin/python

echo "=== TRAIN start $(date '+%F %H:%M:%S') ==="
$PY -u -m scripts.run_saft_pelvic_retrain
echo "=== TRAIN done $(date '+%F %H:%M:%S') ==="

echo "=== EVAL start $(date '+%F %H:%M:%S') ==="
$PY -u -m scripts.eval_saft_heldout
echo "=== EVAL done $(date '+%F %H:%M:%S') ==="
