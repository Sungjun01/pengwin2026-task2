#!/usr/bin/env bash
# Dataset021 (FS-DF femur) 전체 빌드 → plan/preprocess → ResEnc-M fold_all 학습 (GPU8).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/d021_train_chain.log
: > "$L"

echo "[chain] D021 femur full build $(date +%H:%M:%S)" >> "$L"
python3 -u -m scripts.build_d021_fsdf_femur >> "$L" 2>&1
grep -q "D021_BUILD_DONE" "$L" || { echo "[chain] build 실패" >> "$L"; exit 2; }

echo "[chain] plan+preprocess $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_and_preprocess -d 21 -c 3d_fullres --verify_dataset_integrity -np 12 >> "$L" 2>&1
echo "[chain] plan ResEnc-M $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_experiment -d 21 -pl nnUNetPlannerResEncM >> "$L" 2>&1
echo "[chain] === train ResEnc-M fold_all (GPU8) $(date +%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=8 nnUNetv2_train 21 3d_fullres all -p nnUNetResEncUNetMPlans >> "$L" 2>&1
echo "D021_TRAIN_DONE rc=$? $(date +%H:%M:%S)" >> "$L"
