#!/usr/bin/env bash
# Dataset020 (FS-DF graded) 전체 빌드 → plan/preprocess → ResEnc-M fold_all 학습 (GPU7).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/d020_train_chain.log
: > "$L"

echo "[chain] D020 full build $(date +%H:%M:%S)" >> "$L"
python3 -m scripts.build_d020_fsdf >> "$L" 2>&1
grep -q "D020_BUILD_DONE" "$L" || { echo "[chain] build 실패" >> "$L"; exit 2; }

echo "[chain] plan+preprocess $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_and_preprocess -d 20 -c 3d_fullres --verify_dataset_integrity -np 12 >> "$L" 2>&1
echo "[chain] plan ResEnc-M $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_experiment -d 20 -pl nnUNetPlannerResEncM >> "$L" 2>&1
echo "[chain] === train ResEnc-M fold_all (GPU7) $(date +%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 20 3d_fullres all -p nnUNetResEncUNetMPlans >> "$L" 2>&1
echo "D020_TRAIN_DONE rc=$? $(date +%H:%M:%S)" >> "$L"
