#!/usr/bin/env bash
# Task F (Dataset008 Spatial Moments) 1000 epoch 자연 완료 후 V1 Oracle (Dataset011) on GPU 6 자동 launch.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/nnunet_env.sh"

TASK_F_FINAL="$nnUNet_results/Dataset008_PENGWIN_SpatialMoments/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"
echo "[chain V1-Oracle] waiting for Task F checkpoint_final at $TASK_F_FINAL"
until [[ -f "$TASK_F_FINAL" ]]; do
    sleep 60
done
echo "[chain V1-Oracle] Task F finished — launching V1 Oracle (D011) on GPU 6"
exec env CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 11 3d_fullres all
