#!/usr/bin/env bash
# Autonomous SDT femur pipeline: wait for the dataset build -> preprocess -> train fold_0.
# Standard nnUNet (11-class SDT energy bins) = no custom trainer, no warm-start (no hang).
# Arg1 = GPU index (default 3). Log: logs/sdt_pipeline.log
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU=${1:-3}
P=logs/sdt_pipeline.log; mkdir -p logs; : > "$P"

echo "[sdt-pipe] $(date +%F_%H:%M:%S) waiting for dataset build (SDT_BUILD_DONE)..." >> "$P"
until grep -q "SDT_BUILD_DONE" logs/build_sdt_femur.log 2>/dev/null; do
  if ! pgrep -f build_sdt_femur >/dev/null 2>&1; then
    grep -q "SDT_BUILD_DONE" logs/build_sdt_femur.log 2>/dev/null || { echo "[sdt-pipe] BUILD DIED w/o DONE — abort" >> "$P"; exit 1; }
  fi
  sleep 20
done
echo "[sdt-pipe] $(date +%F_%H:%M:%S) build done. plan+preprocess Dataset023..." >> "$P"

nnUNetv2_plan_and_preprocess -d 23 --verify_dataset_integrity -np 8 >> "$P" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[sdt-pipe] PREPROCESS FAILED rc=$rc — abort (no GPU wasted)" >> "$P"; exit $rc; fi
echo "[sdt-pipe] $(date +%F_%H:%M:%S) preprocess done. train fold_0 on GPU$GPU..." >> "$P"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_train 23 3d_fullres 0 >> "$P" 2>&1
echo "SDT_PIPELINE_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$P"
