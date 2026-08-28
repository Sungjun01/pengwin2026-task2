#!/usr/bin/env bash
# D019 (predicted-channel, mismatch-fixed base) + BorderTopo_b05 (topology loss,
# beta=0.05) + ResEnc-L (24GB max capacity, patch 160x160x256). The best-model build:
# correct base + separation root-fix + max capacity that fits a 24GB GPU.
#   arg1 = fold (0 | all),  arg2 = CUDA device index,  arg3 = plans (default ResEnc-M)
set -uo pipefail
FOLD="${1:?fold}"
GPU="${2:?gpu}"
PLANS="${3:-nnUNetResEncUNetMPlans}"
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L="logs/topo_d019_${PLANS}_fold${FOLD}_train.log"
mkdir -p logs
: > "$L"
echo "[d019] === D019 BorderTopo_b05 ${PLANS} fold ${FOLD} (GPU${GPU}) $(date +%F_%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 19 3d_fullres "$FOLD" \
  -tr nnUNetTrainerBorderTopo_b05 -p "$PLANS" >> "$L" 2>&1
echo "TOPO_D019_${PLANS}_FOLD${FOLD}_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
