#!/usr/bin/env bash
# fold_all 이 ep250 도달하면 자동으로 held-out 6케이스 predict + instance 채점.
# 학습 죽거나 에러나면 중단하고 보고. GPU1(free) 사용 — fold_all(GPU0)과 충돌 없음.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
LOG=logs/resencM_d015/fold_all.log
PROC="train 15 3d_fullres all -p nnUNetResEncUNetMPlans"

echo "[trigger] waiting for Epoch 250 in $LOG ..."
while true; do
  if grep -qE "Epoch 250\b" "$LOG" 2>/dev/null; then echo "[trigger] EP250 reached"; break; fi
  if grep -qE "Traceback|RuntimeError|CUDA error|out of memory" "$LOG" 2>/dev/null; then echo "[trigger] TRAINING ERROR detected — aborting"; exit 2; fi
  if ! pgrep -f "$PROC" >/dev/null 2>&1; then echo "[trigger] TRAINING PROCESS GONE — aborting"; exit 3; fi
  sleep 60
done

source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=8
OUT=predictions/resencM_d015_ep250
mkdir -p "$OUT"
echo "[trigger] === predict 6 held-out cases (checkpoint_best, GPU1) ==="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i predictions/d015_heldout_in -o "$OUT" \
  -d 15 -c 3d_fullres -p nnUNetResEncUNetMPlans \
  -tr nnUNetTrainerBorderWeightedAseg_w40 -f all -chk checkpoint_best.pth \
  2>&1 | tail -5
echo "[trigger] === score @ min_vol {200,100,50} (post-proc curve) vs gt_instance ==="
for mv in 200 100 50; do
  echo "########## resencM_ep250_mv${mv} ##########"
  python3 -m scripts.eval_heldout_bordercore \
    --pred_dir "$OUT" --gt_dir predictions/gt_instance \
    --tag resencM_ep250_mv${mv} --min_vol $mv --mode legacy --work /tmp/pengwin_resenc 2>/dev/null \
    | grep -E "LDSC_F_mean|RQ_F_mean|HD95_F_mm_mean|IoU_F_mean"
done
echo "EP250_EVAL_DONE"
