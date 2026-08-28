#!/usr/bin/env bash
# Task A (Dataset007 vanilla, GPU 5) 학습이 끝나는 즉시 Task D (Dataset009 Oracle) 학습을
# GPU 5 에서 launch. Task A 가 1000 epoch 자연 완료 또는 종료될 때까지 30초 간격으로 polling.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ"

echo "[chain] waiting for Task A (Dataset007 vanilla) to finish on GPU 5..."
while pgrep -f "nnUNetv2_train 7 3d_fullres all" > /dev/null 2>&1; do
    sleep 30
done

echo "[chain] Task A finished — launching Task D (Dataset009 Oracle) on GPU 5"
# shellcheck disable=SC1091
source scripts/nnunet_env.sh
exec env CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 9 3d_fullres all
