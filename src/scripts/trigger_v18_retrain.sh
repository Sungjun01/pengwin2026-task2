#!/usr/bin/env bash
# v18 champion recipe (BorderWeightedAseg_w40) from-scratch + 50-ep snapshots (+ best).
# Goal: find the instance-best epoch — 1000ep is not always best (overfit). Dataset016 fold_0.
# Arg1 = GPU (default 2). PCI_BUS_ID ordering.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export V18_SNAPSHOT_EVERY=50
GPU=${1:-2}
L=logs/v18_retrain.log; mkdir -p logs; : > "$L"
echo "[v18] champion recipe from-scratch Dataset016 fold0 GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
echo "[v18] 1000ep, 50-ep snapshots + best — pick instance-best epoch (overfit guard)" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerV18Snap >> "$L" 2>&1
echo "V18_RETRAIN_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
