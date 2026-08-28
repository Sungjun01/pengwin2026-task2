#!/bin/bash
# Wait for the SAFT retrain to finish, then evaluate the held-out hard cases.
set -u
REPO=/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
cd "$REPO" || exit 1
LOG=logs/saft_pelvic_fixed_f48_p96/run.log
HIST=logs/saft_pelvic_fixed_f48_p96/history.json

echo "=== waiter started $(date '+%H:%M:%S') ==="
for i in $(seq 1 240); do          # up to 240 min
  if grep -q '"event": "retrain_done"' "$LOG" 2>/dev/null; then
    echo "=== retrain_done detected at iter $i ($(date '+%H:%M:%S')) ==="
    break
  fi
  # stall guard: history.json should update every epoch
  if [ -f "$HIST" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HIST") ))
    if [ "$age" -gt 1200 ]; then
      echo "=== STALL: history.json unmodified for ${age}s, proceeding to eval with best.pt ==="
      break
    fi
  fi
  sleep 60
done

echo "=== final training tail ==="
tail -2 "$LOG" 2>/dev/null

if [ ! -f logs/saft_pelvic_fixed_f48_p96/best.pt ]; then
  echo "=== NO best.pt -- aborting eval ==="
  exit 2
fi

echo "=== running held-out eval $(date '+%H:%M:%S') ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" \
  .venv_voxel_mamba/bin/python -u -m scripts.eval_saft_heldout
echo "=== waiter+eval finished $(date '+%H:%M:%S') ==="
