#!/usr/bin/env bash
# Finalize V44 (HQ-lite): authoritative gate on the ACTUAL v44 image (baked env, not runtime override)
# + robust tar.gz save (concurrent: inference on GPU, save on CPU).
set -u
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
LOG=logs/finalize_v44.log
: > "$LOG"
ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

IN=/input/images/peripelvic-fracture-ct
OUT=/output/images/peripelvic-fracture-ct-segmentation

# --- (A) save v44 -> tar.gz (CPU, concurrent), .tmp -> verify -> atomic rename ---
( OUTF=pengwin-task1-v44-hqlite.tar.gz; TMP="$OUTF.tmp"
  echo "[$(ts)] SAVE start -> $TMP" >> "$LOG"
  rm -f "$TMP"
  if docker save pengwin-task1-v44 | pigz -p 24 > "$TMP" && pigz -t "$TMP" 2>>"$LOG"; then
    mv -f "$TMP" "$OUTF"; echo "[$(ts)] SAVE OK $OUTF ($(du -h "$OUTF"|cut -f1))" >> "$LOG"
  else echo "[$(ts)] SAVE FAIL (left $TMP)" >> "$LOG"; fi ) &
SAVE_PID=$!

# --- (B) authoritative gate: run ACTUAL v44 image on public-5 (GPU1) ---
rm -rf /tmp/v44_out && mkdir -p /tmp/v44_out && chmod 777 /tmp/v44_out
say "=== v44 image on public-5 (GPU1) — confirm baked env + gate ==="
docker run --rm --gpus '"device=1"' \
  -v /tmp/pub5_in:$IN -v /tmp/v44_out:$OUT \
  pengwin-task1-v44:latest > /tmp/v44_run.log 2>&1
say "v44 run rc=$?"
say "--- baked env confirm (must show pelvic TTA=False) ---"
grep -E "HQ inference|\[done\].*elapsed|TTA enabled" /tmp/v44_run.log | tee -a "$LOG"

say "=== proxy-GT gate on ACTUAL v44 outputs ==="
PYTHONPATH=. python3 -m scripts.proxy_gt_eval --cand_dir /tmp/v44_out 2>&1 \
  | grep -vE "FutureWarning|warn_deprecated|deprecate" | tee -a "$LOG"

say "=== waiting for tar.gz save ==="
wait $SAVE_PID
tail -2 "$LOG" >/dev/null
grep -E "SAVE (OK|FAIL)" "$LOG" | tail -1
say "=== FINALIZE_V44_DONE ==="
