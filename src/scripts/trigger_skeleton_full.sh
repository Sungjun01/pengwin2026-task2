#!/usr/bin/env bash
# FULL from-scratch Skeleton Recall (Dataset016 fold_0, held-out). 1000ep / lr 1e-2 / warmup 100.
# Keeps 50-ep snapshots (checkpoint_ep50/100/...) + checkpoint_best — 1000ep is not always best
# (overfit), so we gate the trajectory and pick the instance-best epoch. Arg1 = GPU (default 1).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SKEL_LAMBDA=0.5 SKEL_DILATE=1 SKEL_EVERY=4 SKEL_WARMUP=100 SKEL_SNAPSHOT_EVERY=50
# (unset) SKEL_EPOCHS -> 1000 default ; (unset) SKEL_LR -> 1e-2 default (from-scratch)
GPU=${1:-1}
L=logs/skeleton_full.log; mkdir -p logs; : > "$L"
echo "[skel-full] from-scratch Skeleton Recall Dataset016 fold0 GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
echo "[skel-full] 1000ep lr1e-2 warmup100 snapshot50 (+best) — gate trajectory, pick best epoch" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerSkeletonRecallFull >> "$L" 2>&1
echo "SKELETON_FULL_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
