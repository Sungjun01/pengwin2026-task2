#!/usr/bin/env bash
# RETRAIN FROM SCRATCH (Dataset016 fold_0) with the Supervoxel critical-component
# topology loss — held-out gate for the merge fix (arXiv:2501.01022).
#   - from scratch (no -pretrained_weights): nnUNet default schedule (1000 ep, lr 1e-2)
#   - SV_WARMUP=100: base loss only for the first 100 ep (predictions are garbage early,
#     so the topology term would chase spurious bridges); supervoxel term turns on after.
#   - persistent snapshot every 50 ep -> checkpoint_ep50/100/...pth (gate-eval the trajectory;
#     V20 lesson: val-dice "best" != separation-optimal, so keep snapshots to score each).
# GPU5 = RTX 4090 D; MUST set CUDA_DEVICE_ORDER=PCI_BUS_ID (memory: 4090 ordering trap).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # avoid thread oversubscription across parallel trainings

# Supervoxel loss knobs (explicit w_merge/w_split; merge is the main enemy). SV_EVERY=4: the
# critical-component CC is CPU-bound, so run the SV term every 4 steps (4x less CC -> GPU not starved).
export SV_LAMBDA=0.5 SV_W_MERGE=1.0 SV_W_SPLIT=0.25 SV_MIN_SIZE=2 SV_EVERY=4
export SV_WARMUP=100 SV_SNAPSHOT_EVERY=50
# (unset) SV_EPOCHS -> 1000 default ;  (unset) SV_LR -> 1e-2 default (scratch)

L=logs/supervoxel_fold0_scratch.log
mkdir -p logs
: > "$L"

echo "[sv] === RETRAIN-SCRATCH Supervoxel Dataset016 3d_fullres fold 0 on GPU5(4090) $(date +%F_%H:%M:%S) ===" >> "$L"
echo "[sv] lambda=$SV_LAMBDA w_merge=$SV_W_MERGE w_split=$SV_W_SPLIT warmup=$SV_WARMUP snapshot_every=$SV_SNAPSHOT_EVERY (epochs=1000 lr=1e-2 default)" >> "$L"
GPU=${1:-5}
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 \
  -tr nnUNetTrainerSupervoxelTopo >> "$L" 2>&1
echo "SUPERVOXEL_FOLD0_SCRATCH_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
