#!/usr/bin/env bash
# Fast warm-start PROBE of the BoRT loss on a FREE GPU (default GPU1, a 3090). Fine-tune the
# champion fold_0 with the BoRT term active from epoch 0 -> isolates the skeleton-aware
# narrow-seam weighting effect at champion base quality. Parallel to the Supervoxel probe;
# separate results folder (nnUNetTrainerBoRT). Same fold_0 held-out gate (run_fold0_gate.py).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # avoid thread oversubscription across parallel trainings
export BORT_LAMBDA=0.5 BORT_W0=10.0 BORT_TOPO=0.1 BORT_WMAX_MM=8.0
export BORT_WARMUP=0 BORT_SNAPSHOT_EVERY=20 BORT_EPOCHS=60 BORT_LR=1e-3
GPU=${1:-1}
CKPT="$nnUNet_results/Dataset016_PENGWIN_BorderCore_MedialOrient/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
L=logs/bort_probe_fold0.log; mkdir -p logs; : > "$L"
echo "[bort] warm-start BoRT Dataset016 fold0 on GPU$GPU $(date +%F_%H:%M:%S)" >> "$L"
echo "[bort] ckpt=$CKPT  warmup=0 lr=1e-3 epochs=60 snapshot_every=20 lambda=0.5 w0=10" >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 16 3d_fullres 0 -tr nnUNetTrainerBoRT \
  -pretrained_weights "$CKPT" >> "$L" 2>&1
echo "BORT_PROBE_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
