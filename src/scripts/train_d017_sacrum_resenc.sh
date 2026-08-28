#!/usr/bin/env bash
# D017: LPS sacrum binary + ResEnc-M (replaces D006 Stage-1 backbone).
set -euo pipefail
ROOT="/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
source "${ROOT}/scripts/nnunet_env.sh"

GPU="${1:-4}"
LOG="${ROOT}/logs/d017_sacrum_resenc_foldall.log"

mkdir -p "${ROOT}/logs"
echo "[D017] GPU=${GPU} log=${LOG}"
exec env CUDA_VISIBLE_DEVICES="${GPU}" \
  nnUNetv2_train 17 3d_fullres all -p nnUNetResEncUNetMPlans \
  2>&1 | tee -a "${LOG}"
