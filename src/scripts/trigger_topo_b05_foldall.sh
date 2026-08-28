#!/usr/bin/env bash
# BorderTopo beta=0.05 fold_all 1000ep (submission candidate, separation root-fix).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/topo_b05_foldall_train.log
mkdir -p logs
: > "$L"
echo "[topo_b05] === train BorderTopo_b05 Dataset016 3d_fullres fold all (GPU2) $(date +%F_%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 16 3d_fullres all \
  -tr nnUNetTrainerBorderTopo_b05 >> "$L" 2>&1
echo "TOPO_B05_FOLDALL_TRAIN_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
