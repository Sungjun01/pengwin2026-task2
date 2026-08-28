#!/usr/bin/env bash
# D018 빌드 완료 → plan/preprocess → ResEnc-M fold_all 학습 자동 체인.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/d018_train_chain.log
BUILD_LOG=logs/build_d018.log
: > "$L"

echo "[chain] waiting for D018 build ($(date +%H:%M:%S)) ..." >> "$L"
while ! grep -q "D018_BUILD_DONE" "$BUILD_LOG" 2>/dev/null; do
  pgrep -f "build_d018_predicted_channels" >/dev/null 2>&1 || { grep -q "D018_BUILD_DONE" "$BUILD_LOG" 2>/dev/null || { echo "[chain] build proc gone w/o DONE — abort" >> "$L"; exit 2; }; }
  sleep 120
done
echo "[chain] build done. plan+preprocess (default) $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_and_preprocess -d 18 -c 3d_fullres --verify_dataset_integrity -np 12 >> "$L" 2>&1
echo "[chain] plan ResEnc-M $(date +%H:%M:%S)" >> "$L"
nnUNetv2_plan_experiment -d 18 -pl nnUNetPlannerResEncM >> "$L" 2>&1
echo "[chain] === train ResEnc-M fold_all (GPU0) $(date +%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 18 3d_fullres all \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerBorderWeightedAseg_w40 >> "$L" 2>&1
echo "D018_TRAIN_DONE rc=$? $(date +%H:%M:%S)" >> "$L"
