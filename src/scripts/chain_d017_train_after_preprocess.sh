#!/usr/bin/env bash
# Wait for D017 preprocess, then launch ResEnc-M fold_all on GPU 4.
set -euo pipefail
ROOT="/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
PREP_LOG="${ROOT}/logs/d017_plan_preprocess.log"
MARKER="${ROOT}/logs/d017_preprocess_done"

source "${ROOT}/scripts/nnunet_env.sh"

PP_DIR="${nnUNet_preprocessed}/Dataset017_PENGWIN_SacrumResEnc_LPS/nnUNetPlans_3d_fullres"
echo "[chain] waiting for 170 preprocessed cases in ${PP_DIR}..."
while true; do
  n=$(find "${PP_DIR}" -maxdepth 1 -name 'PENGWIN_*_seg.b2nd' 2>/dev/null | wc -l)
  if [ "${n}" -ge 170 ]; then
    break
  fi
  sleep 60
done
sleep 30
touch "${MARKER}"
echo "[chain] preprocess done → starting D017 training"
exec bash "${ROOT}/scripts/train_d017_sacrum_resenc.sh" 4
