#!/bin/bash
# Bake the trained all-data SAFT checkpoint into the submission Docker, smoke-test
# on a real case (GPU 1), and save the grand-challenge upload tarball.
set -e
REPO=/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
cd "$REPO"
CKPT=logs/saft_alldata_f48_p96/best.pt
IMG=pengwin-saft:alldata-f48-p96
TAR=pengwin-task1-saft.tar.gz

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
cp "$CKPT" saft_model/best.pt
echo "=== baked model: $(du -h saft_model/best.pt | cut -f1) ==="

echo "=== docker build ==="
DOCKER_BUILDKIT=1 docker build -f Dockerfile.saft -t "$IMG" .

echo "=== real-case container smoke (GPU 1) ==="
T=$(mktemp -d /tmp/saft_realsmoke_XXXX)
mkdir -p "$T/input/images/peripelvic-fracture-ct" "$T/output"; chmod -R 777 "$T"
# copy one real pelvic case (004) into GC input layout
cp "$(find /home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset -path '*/004/image.mha')" \
   "$T/input/images/peripelvic-fracture-ct/case004.mha"
docker run --rm --gpus '"device=1"' -v "$T/input:/input:ro" -v "$T/output:/output" "$IMG"
.venv_voxel_mamba/bin/python - "$T" <<'PY'
import sys, SimpleITK as sitk, numpy as np
T=sys.argv[1]
a=sitk.GetArrayFromImage(sitk.ReadImage(f"{T}/output/images/peripelvic-fracture-ct-segmentation/case004.mha"))
ids=sorted(int(x) for x in set(np.unique(a))-{0})
print(f"REAL-CASE 004 container output: n_fragments={len(ids)} ids={ids}")
PY

echo "=== docker save -> $TAR ==="
docker save "$IMG" | gzip > "$TAR"
echo "=== tarball: $(du -h $TAR | cut -f1) ==="
echo "=== PACKAGE DONE ==="
