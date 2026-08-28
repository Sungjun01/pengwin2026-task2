#!/usr/bin/env bash
# Supervoxel loss, fold_all (SUBMISSION model — trains on all 170, no held-out). The fold_0
# run is the gate; this is the deployable model the gate's verdict applies to. From-scratch
# (reliable), OMP-capped (avoid thread oversubscription with the other parallel trainings).
# Arg1 = GPU index (default 2). Log: logs/supervoxel_foldall.log
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SV_LAMBDA=0.5 SV_W_MERGE=1.0 SV_W_SPLIT=0.25 SV_MIN_SIZE=2 SV_EVERY=4
export SV_WARMUP=100 SV_SNAPSHOT_EVERY=50
GPU=${1:-2}
L=logs/supervoxel_foldall.log; mkdir -p logs; : > "$L"
echo "[sv-all] === Supervoxel Dataset016 3d_fullres fold ALL on GPU$GPU $(date +%F_%H:%M:%S) ===" >> "$L"
echo "[sv-all] from-scratch, warmup=100, snapshot=50, epochs=1000 lr=1e-2 (submission model)" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres all -tr nnUNetTrainerSupervoxelTopo >> "$L" 2>&1
echo "SUPERVOXEL_FOLDALL_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
