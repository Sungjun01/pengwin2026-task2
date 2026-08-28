#!/usr/bin/env bash
# Re-export already-built docker images to .tar.gz (the build is fine; only the
# earlier `docker save | gzip` was interrupted -> truncated tars). Write to .tmp,
# verify with `pigz -t`, then atomically rename. Idempotent & safe to re-run.
set -o pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
LOG=logs/resave_v21_v43.log
mkdir -p logs
ts() { date '+%F %T'; }

save_one() {
  local image="$1" out="$2"
  local tmp="${out}.tmp"
  echo "[$(ts)] START  $image -> $tmp" | tee -a "$LOG"
  rm -f "$tmp"
  docker save "$image" | pigz -p 24 > "$tmp"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(ts)] FAIL   $image  (pipe rc=$rc) -- leaving $tmp for inspection" | tee -a "$LOG"
    return 1
  fi
  echo "[$(ts)] VERIFY $tmp" | tee -a "$LOG"
  if pigz -t "$tmp" 2>>"$LOG"; then
    mv -f "$tmp" "$out"
    echo "[$(ts)] OK     $out  ($(du -h "$out" | cut -f1))" | tee -a "$LOG"
    return 0
  else
    echo "[$(ts)] FAIL   $tmp failed integrity test -- NOT renaming" | tee -a "$LOG"
    return 1
  fi
}

echo "[$(ts)] ==== resave start ====" | tee -a "$LOG"
save_one pengwin-task1-v21 pengwin-task1-v21-hq.tar.gz       && \
save_one pengwin-task1-v43 pengwin-task1-v43-hq-5fold.tar.gz
RC=$?
echo "[$(ts)] ==== resave done (rc=$RC) ====" | tee -a "$LOG"
exit $RC
