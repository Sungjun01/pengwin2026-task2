#!/usr/bin/env bash
# 학습 완료(checkpoint_final.pth) 시 자동 최종 held-out 평가.
# 완료 후엔 GPU/CPU 비므로 경합 없이 깨끗이 predict+채점.
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
RD=/home/schoaq/taehee/nnUNet_data/results/Dataset015_PENGWIN_BorderCore_OrientFixed/nnUNetTrainerBorderWeightedAseg_w40__nnUNetResEncUNetMPlans__3d_fullres/fold_all
PROC="python3 .*/nnUNetv2_train 15 3d_fullres all"
ELOG=logs/resencM_d015/eval_final.log
: > "$ELOG"

echo "[ep999] waiting for $RD/checkpoint_final.pth ..." >> "$ELOG"
while [ ! -f "$RD/checkpoint_final.pth" ]; do
  if ! pgrep -f "$PROC" >/dev/null 2>&1; then
    sleep 5
    [ -f "$RD/checkpoint_final.pth" ] && break
    echo "[ep999] TRAINING DIED before checkpoint_final — aborting" >> "$ELOG"; exit 3
  fi
  sleep 120
done
echo "[ep999] TRAINING COMPLETE (checkpoint_final present)" >> "$ELOG"

source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
OUT=predictions/resencM_d015_final
mkdir -p "$OUT"
for CHK in checkpoint_best checkpoint_final; do
  O="${OUT}_${CHK}"; mkdir -p "$O"
  echo "[ep999] === predict ($CHK, GPU0) ===" >> "$ELOG"
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i predictions/d015_heldout_in -o "$O" \
    -d 15 -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainerBorderWeightedAseg_w40 \
    -f all -chk ${CHK}.pth -npp 4 -nps 4 >> "$ELOG" 2>&1
  for mv in 100 200 500 1000; do
    echo "########## ${CHK}_mv${mv} ##########" >> "$ELOG"
    python3 -m scripts.eval_heldout_bordercore --pred_dir "$O" --gt_dir predictions/gt_instance \
      --tag final_${CHK}_mv${mv} --min_vol $mv --mode legacy --work /tmp/pengwin_final 2>/dev/null \
      | grep -E "LDSC_F_mean|RQ_F_mean|HD95_F_mm_mean|IoU_F_mean|PENGWIN_" >> "$ELOG"
  done
done
echo "EP999_EVAL_DONE" >> "$ELOG"
