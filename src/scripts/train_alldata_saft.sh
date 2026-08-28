#!/bin/bash
# Train SAFT on ALL 340 PENGWIN cases (pelvic + femur), fixed pipeline, for the Docker submission model.
set -e
REPO=/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO"
PY=.venv_voxel_mamba/bin/python
TR=/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset
OUT=logs/saft_alldata_f48_p96

echo "=== TRAIN all-340 start $(date '+%F %H:%M:%S') ==="
$PY -u -m scripts.train_saft \
  --raw-root "$TR" --streaming --cache-size 340 \
  --feature-channels 48 --patch-size 96,96,96 \
  --epochs 150 --steps-per-epoch 100 --lr 2e-4 \
  --device cuda --output-dir "$OUT"
echo "=== TRAIN done $(date '+%F %H:%M:%S') ==="

echo "=== smoke infer (sanity: model not collapsed) ==="
$PY - <<'PYEOF'
import numpy as np
from scripts.voxel_mamba_dataset import discover_raw_cases
from scripts.infer_saft_case import infer_saft_case_from_paths
from scripts.voxel_mamba_io import load_lps_volume
ck="logs/saft_alldata_f48_p96/best.pt"
cases={c[0]:c for c in discover_raw_cases("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset")}
for cid in ("001","300"):
    img=cases[cid][1]
    out=f"logs/saft_alldata_f48_p96/smoke_{cid}.mha"
    infer_saft_case_from_paths(ck, img, out, min_voxels=20, device="cuda", decode="affinity")
    pred=load_lps_volume(out, dtype=np.int32).array
    ids=sorted(int(v) for v in set(np.unique(pred))-{0})
    print(f"SMOKE case {cid}: n_pred_fragments={len(ids)} ids={ids[:8]}{'...' if len(ids)>8 else ''}", flush=True)
PYEOF
echo "=== all done $(date '+%F %H:%M:%S') ==="
