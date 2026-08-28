#!/usr/bin/env bash
# Build the PENGWIN 2026 Task 2 submission container.
#
#   ./build.sh              -> builds image  pengwin-task2-v75:latest
#   ./build.sh --save       -> also writes   pengwin-task2-v75.tar.gz  (Grand Challenge upload)
#
# Requires model weights under ./models/ — run ./fetch_weights.sh first.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-pengwin-task2-v75}"
TAG="${TAG:-latest}"

if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "ERROR: ./models/ is empty. Download the weights first:" >&2
    echo "         ./fetch_weights.sh" >&2
    echo "       (see README.md -> Model weights)" >&2
    exit 1
fi

echo "[build] docker build -t ${IMAGE}:${TAG} ."
DOCKER_BUILDKIT=1 docker build -t "${IMAGE}:${TAG}" .

if [ "${1:-}" = "--save" ]; then
    OUT="${IMAGE}.tar.gz"
    echo "[save] ${OUT}"
    if command -v pigz >/dev/null 2>&1; then
        docker save "${IMAGE}:${TAG}" | pigz -p "$(nproc)" > "${OUT}"
        pigz -t "${OUT}"
    else
        docker save "${IMAGE}:${TAG}" | gzip > "${OUT}"
        gzip -t "${OUT}"
    fi
    ls -lh "${OUT}"
fi

echo "[done] ${IMAGE}:${TAG}"
