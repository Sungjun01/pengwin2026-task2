#!/usr/bin/env bash
# Measure HQ-lite (pelvic TTA OFF, gaussian+0.5 overlap, fold_all; femur full HQ) timing
# WITHOUT rebuilding: run the v21 image with -e PIVOT_PELVIC_TTA=0 override.
# Also run v18/v19 baseline (known to fit T4) on the heaviest pelvic case for calibration.
# Then run proxy-GT gate on the HQ-lite public-5 outputs.
set -u
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
LOG=logs/measure_hqlite.log
: > "$LOG"
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

IN_MNT=/input/images/peripelvic-fracture-ct
OUT_MNT=/output/images/peripelvic-fracture-ct-segmentation

# inputs already staged at /tmp/pub5_in/00X.mha (from earlier run)
mkdir -p /tmp/base_in && cp -f /tmp/pub5_in/003.mha /tmp/base_in/003.mha
rm -rf /tmp/hqlite_out /tmp/base003_out
mkdir -p /tmp/hqlite_out /tmp/base003_out && chmod 777 /tmp/hqlite_out /tmp/base003_out

say "=== (1) v18/v19 baseline on 003 (GPU3) — calibration (known to FIT T4) ==="
( docker run --rm --gpus '"device=3"' \
    -v /tmp/base_in:$IN_MNT -v /tmp/base003_out:$OUT_MNT \
    pengwin-task1-v19:latest > /tmp/base003.log 2>&1
  echo "baseline rc=$? $(ts)" >> "$LOG" ) &
BASE_PID=$!

say "=== (2) HQ-lite on public-5 (GPU1) — pelvic TTA OFF override; timing + gate inputs ==="
docker run --rm --gpus '"device=1"' \
  -e PIVOT_PELVIC_TTA=0 \
  -v /tmp/pub5_in:$IN_MNT -v /tmp/hqlite_out:$OUT_MNT \
  pengwin-task1-v21:latest > /tmp/hqlite.log 2>&1
say "HQ-lite rc=$? done"

wait $BASE_PID

say "=== per-case elapsed (HQ-lite) ==="
grep -E "\[done\].*elapsed|HQ inference|folds=" /tmp/hqlite.log | tee -a "$LOG"
say "=== baseline 003 elapsed ==="
grep -E "\[done\].*elapsed|HQ inference" /tmp/base003.log | tee -a "$LOG"

say "=== HQ-lite outputs (instance dist) ==="
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import glob, SimpleITK as sitk, numpy as np
for f in sorted(glob.glob("/tmp/hqlite_out/*.mha")):
    a=sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(int)
    u=[int(v) for v in np.unique(a) if v>0]
    def c(lo,hi): return len([v for v in u if lo<=v<=hi])
    print(f"{f.split('/')[-1]:10s} sac={c(1,50)} LH={c(51,100)} RH={c(101,150)} fem={c(151,200)}")
PY

say "=== proxy-GT gate: HQ-lite vs v18 (002/004/005=재현?, 001/003=구조) ==="
PYTHONPATH=. python3 -m scripts.proxy_gt_eval --cand_dir /tmp/hqlite_out 2>&1 | grep -vE "FutureWarning|warn_deprecated|deprecate" | tee -a "$LOG"
say "=== MEASURE_HQLITE_DONE ==="
