#!/usr/bin/env bash
# Quick status check for ongoing nnUNet training runs.
# Usage: bash scripts/check_training.sh

echo "=== Process Status ==="
pgrep -af "nnUNetv2_train 00" | grep -v bash | head -5 || echo "NOT RUNNING"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo
echo "=== D001 (anatomical) latest epoch ==="
grep -E "^[0-9-]+ [0-9:.]+: Epoch [0-9]+$|^[0-9-]+ [0-9:.]+: Mean Validation" /tmp/train_001_full.log 2>/dev/null | tail -5

echo
echo "=== D002 (CSM) latest epoch ==="
grep -E "^[0-9-]+ [0-9:.]+: Epoch [0-9]+$|^[0-9-]+ [0-9:.]+: Mean Validation" /tmp/train_002_full.log 2>/dev/null | tail -5

echo
echo "=== Checkpoints ==="
for d in 001:Anatomical 002:Frac; do
    id=${d%%:*}; name=${d##*:}
    dir="/home/schoaq/taehee/nnUNet_data/results/Dataset${id}_PENGWIN_${name}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"
    if [[ -d "$dir" ]]; then
        echo "D${id}: $(ls -t "$dir"/checkpoint*.pth 2>/dev/null | head -2 | xargs -I{} basename {} | tr '\n' ' ')"
    else
        echo "D${id}: no checkpoint dir yet"
    fi
done
