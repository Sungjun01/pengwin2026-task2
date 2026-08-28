#!/usr/bin/env bash
# RETRAIN FROM SCRATCH (Dataset016 fold_0) with the BoRT loss. From-scratch is the RELIABLE
# path here (warm-start -pretrained_weights hangs in nnUNet get_dataloaders on this loaded box).
# Same recipe as the Supervoxel scratch: 1000 ep / lr 1e-2 / warmup 100 / snapshot every 50.
# PCI_BUS_ID ordering (memory: 4090/ordering trap). Arg1 = GPU index (default 1).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=10
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BORT_LAMBDA=0.5 BORT_W0=10.0 BORT_TOPO=0.1 BORT_WMAX_MM=8.0
export BORT_WARMUP=100 BORT_SNAPSHOT_EVERY=50
# (unset) BORT_EPOCHS -> 1000 ;  (unset) BORT_LR -> 1e-2 (scratch)
GPU=${1:-1}
L=logs/bort_scratch.log; mkdir -p logs; : > "$L"
echo "[bort] === RETRAIN-SCRATCH BoRT Dataset016 fold0 on GPU$GPU $(date +%F_%H:%M:%S) ===" >> "$L"
echo "[bort] lambda=0.5 w0=10 topo=0.1 warmup=100 snapshot=50 (epochs=1000 lr=1e-2 default)" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerBoRT >> "$L" 2>&1
echo "BORT_SCRATCH_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
