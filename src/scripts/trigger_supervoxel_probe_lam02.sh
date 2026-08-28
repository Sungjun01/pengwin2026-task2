#!/usr/bin/env bash
# TUNED Supervoxel probe: lambda=0.2 + w_split=0 — reduce the over-correction seen at
# lambda=0.5 (ep20/ep60: merge+split BOTH rose). Lower SV weight + drop the FN/split-pushing
# term -> attack merge only, gentler. Warm-start champion fold_0, SV active from step 0, 60ep.
# Reuses the Probe trainer/folder (old lambda=0.5 results already saved in proxy_out/gate_probe_*).
# Arg1 = GPU (default 5, the 4090). PCI_BUS_ID ordering.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SV_LAMBDA=0.2 SV_W_MERGE=1.0 SV_W_SPLIT=0.0 SV_MIN_SIZE=2 SV_EVERY=4
export SV_WARMUP=0 SV_SNAPSHOT_EVERY=20 SV_EPOCHS=60 SV_LR=1e-3
GPU=${1:-5}
CKPT="$nnUNet_results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
L=logs/supervoxel_probe_lam02.log; mkdir -p logs; : > "$L"
echo "[lam02] lambda=0.2 w_split=0 warm-start fold0 GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerSupervoxelTopoProbe \
  -pretrained_weights "$CKPT" >> "$L" 2>&1
echo "PROBE_LAM02_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
