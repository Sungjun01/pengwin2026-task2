#!/usr/bin/env bash
# D019 (PredChannels retrain): plan+preprocess → ResEnc-M plan → fold_all train.
set -o pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
export PYTHONPATH="${PYTHONPATH:-}"
source scripts/nnunet_env.sh
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
# GPU0 has the most free VRAM (~20GB); other GPUs are used by other users' jobs.
D019_GPU="${D019_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
L=logs/d019_retrain_chain.log

echo "[d019] plan+preprocess 3d_fullres $(date)" | tee -a "$L"
nnUNetv2_plan_and_preprocess -d 19 -c 3d_fullres --verify_dataset_integrity -np 6 >>"$L" 2>&1

echo "[d019] plan ResEnc-M $(date)" | tee -a "$L"
nnUNetv2_plan_experiment -d 19 -pl nnUNetPlannerResEncM >>"$L" 2>&1

echo "[d019] train ResEnc-M fold_all GPU${D019_GPU} $(date)" | tee -a "$L"
nvidia-smi -i "${D019_GPU}" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tee -a "$L"
CUDA_VISIBLE_DEVICES="${D019_GPU}" nnUNetv2_train 19 3d_fullres all \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerBorderWeightedAseg_w40 >>"$L" 2>&1

echo "D019_RETRAIN_DONE rc=$? $(date)" | tee -a "$L"
