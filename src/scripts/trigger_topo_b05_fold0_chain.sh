#!/usr/bin/env bash
# Chain: wait for the fold_0 HQ-inference probe to free GPU3, then train
# BorderTopo_b05 (beta=0.05) on fold_0 — the held-out gate signal that matches
# the beta=0.05 fold_all submission candidate (fold_all has no OOF to gate on).
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export nnUNet_n_proc_DA=12
L=logs/topo_b05_fold0_train.log
mkdir -p logs
: > "$L"

echo "[chain] waiting for HQ probe (HQ_FOLD0_DONE) $(date +%F_%H:%M:%S)" >> "$L"
while ! grep -q "HQ_FOLD0_DONE" logs/oof_fold0_hq.log 2>/dev/null; do
  pgrep -f oof_fold0_hq_infer.py >/dev/null 2>&1 || { grep -q "HQ_FOLD0_DONE" logs/oof_fold0_hq.log 2>/dev/null || { echo "[chain] HQ proc gone w/o DONE — proceed anyway $(date +%F_%H:%M:%S)" >> "$L"; break; }; }
  sleep 60
done
echo "[chain] === train BorderTopo_b05 Dataset016 3d_fullres fold 0 (GPU3) $(date +%F_%H:%M:%S) ===" >> "$L"
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 16 3d_fullres 0 \
  -tr nnUNetTrainerBorderTopo_b05 >> "$L" 2>&1
echo "TOPO_B05_FOLD0_TRAIN_DONE rc=$? $(date +%F_%H:%M:%S)" >> "$L"
