#!/usr/bin/env bash
# Run the container on one case, using the same interface as Grand Challenge.
#
#   ./run_case.sh <input_dir> <output_dir>
#
# <input_dir> must be laid out exactly as Grand Challenge mounts /input:
#
#   <input_dir>/images/peripelvic-fracture-ct/<UUID>.mha
#   <input_dir>/peripelvic-fragment-clicks.json
#
# Result:
#
#   <output_dir>/images/peripelvic-fracture-ct-segmentation/<UUID>.mha
#
# Grand Challenge runs the container with no network, a read-only /input,
# 16 GB host RAM and one GPU; the flags below mirror that.
set -euo pipefail

IN="${1:?usage: ./run_case.sh <input_dir> <output_dir>}"
OUT="${2:?usage: ./run_case.sh <input_dir> <output_dir>}"
IMAGE="${IMAGE:-pengwin-task2-v75}:${TAG:-latest}"

IN="$(readlink -f "$IN")"
mkdir -p "$OUT"; OUT="$(readlink -f "$OUT")"
mkdir -p "$OUT/images/peripelvic-fracture-ct-segmentation"

# The container runs as the unprivileged user `algorithm` (uid inside the image),
# so a host output directory owned by your user is not writable from inside it.
# Grand Challenge mounts a writable /output; locally we have to open it up.
chmod -R 777 "$OUT"

docker run --rm \
    --gpus all \
    --network none \
    --shm-size 4g \
    --memory 16g \
    -v "$IN":/input:ro \
    -v "$OUT":/output \
    "$IMAGE"

echo "[done] output ->"
find "$OUT" -type f
