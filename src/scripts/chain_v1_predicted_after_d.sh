#!/usr/bin/env bash
# Task D (Dataset009 Oracle) 1000 epoch 자연 완료 후 V1 Predicted (Dataset010) on GPU 5 자동 launch.
# checkpoint_final.pth 존재 기반 polling — pgrep stale process 문제 회피.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/nnunet_env.sh"

TASK_D_FINAL="$nnUNet_results/Dataset009_PENGWIN_OracleSacrum/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth"
echo "[chain V1-Predicted] waiting for Task D checkpoint_final at $TASK_D_FINAL"
until [[ -f "$TASK_D_FINAL" ]]; do
    sleep 60
done
echo "[chain V1-Predicted] Task D finished — launching V1 Predicted (D010) on GPU 5"
exec env CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 10 3d_fullres all
