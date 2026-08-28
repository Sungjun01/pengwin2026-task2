#!/usr/bin/env bash
# Official Skeleton Recall (MIC-DKFZ, ECCV'24) on the border seam channel — warm-start PROBE.
# Champion fold_0 fine-tune, SKEL term active from step 0, 60ep, SKEL_EVERY=4 (skeletonize=CPU).
# Gate vs champion to definitively bury or revive (the seam-recall family has a graveyard).
# Arg1 = GPU (default 0). PCI_BUS_ID ordering.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SKEL_LAMBDA=0.5 SKEL_DILATE=1 SKEL_EVERY=4 SKEL_WARMUP=0
export SKEL_SNAPSHOT_EVERY=20 SKEL_EPOCHS=60 SKEL_LR=1e-3
GPU=${1:-0}
CKPT="$nnUNet_results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
L=logs/skeleton_probe.log; mkdir -p logs; : > "$L"
echo "[skel] Skeleton Recall warm-start fold0 GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerSkeletonRecall \
  -pretrained_weights "$CKPT" >> "$L" 2>&1
echo "SKELETON_PROBE_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
