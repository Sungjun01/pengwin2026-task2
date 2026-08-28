#!/bin/bash
REPO=/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
cd "$REPO"
LOG=logs/saft_alldata_f48_p96/pipeline.log
for i in $(seq 1 360); do
  if grep -q "=== all done ===" "$LOG" 2>/dev/null; then echo "TRAIN+SMOKE COMPLETE (iter $i)"; break; fi
  if [ -f "$LOG" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt 1800 ]; then echo "STALL: pipeline.log idle ${age}s -- likely crashed/finished"; break; fi
  fi
  sleep 60
done
echo "=== pipeline.log tail ==="; tail -10 "$LOG" 2>/dev/null
