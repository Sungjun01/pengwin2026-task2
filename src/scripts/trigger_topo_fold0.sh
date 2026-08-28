#!/usr/bin/env bash
# BorderTopo fold_0 ablation (separation root-fix, design 2026-06-10).
# Same dataset/plans/decoder as the OOF170 Aseg_w40 baseline; only the loss adds
# clDice border-continuity + seam-core terms. Compared at final vs Aseg_w40 fold_0.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/topo_fold0_train.log
mkdir -p logs
: > "$L"
echo "[topo] === train BorderTopo Dataset016 3d_fullres fold 0 (GPU0) $(date +%F_%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 16 3d_fullres 0 \
  -tr nnUNetTrainerBorderTopo >> "$L" 2>&1
echo "TOPO_FOLD0_TRAIN_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
