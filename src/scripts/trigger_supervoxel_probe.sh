#!/usr/bin/env bash
# Fast warm-start PROBE of the Supervoxel loss on a FREE GPU (default GPU2, a 3090).
# Fine-tune the champion fold_0 with the SV term active from epoch 0 -> isolates the SV
# effect at champion base quality for a clean, fast held-out gate (parallel to the scratch
# deliverable on GPU5; separate results folder via the Probe trainer subclass).
# PCI_BUS_ID ordering so CUDA index == nvidia-smi index (memory: 4090/ordering trap).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # avoid thread oversubscription across parallel trainings
export SV_LAMBDA=0.5 SV_W_MERGE=1.0 SV_W_SPLIT=0.25 SV_MIN_SIZE=2 SV_EVERY=4
export SV_WARMUP=0 SV_SNAPSHOT_EVERY=20 SV_EPOCHS=60 SV_LR=1e-3
GPU=${1:-2}
CKPT="$nnUNet_results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
L=logs/supervoxel_probe_fold0.log; mkdir -p logs; : > "$L"
echo "[probe] warm-start Supervoxel Dataset016 fold0 on GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
echo "[probe] ckpt=$CKPT  warmup=0 lr=1e-3 epochs=60 snapshot_every=20 lambda=0.5" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerSupervoxelTopoProbe \
  -pretrained_weights "$CKPT" >> "$L" 2>&1
echo "SUPERVOXEL_PROBE_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
