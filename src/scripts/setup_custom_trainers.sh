#!/usr/bin/env bash
# Symlink repo custom nnUNet trainers into the nnUNet install so nnUNetv2_train can discover them.
set -e
NNV="/home/schoaq/.local/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants"
REPO="/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeighted50.py" "$NNV/nnUNetTrainerBorderWeighted50.py"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeighted.py" "$NNV/nnUNetTrainerBorderWeighted.py"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeightedAseg.py" "$NNV/nnUNetTrainerBorderWeightedAseg.py"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeightedAseg_w40.py" "$NNV/nnUNetTrainerBorderWeightedAseg_w40.py"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeightedAseg_w40_cosine.py" "$NNV/nnUNetTrainerBorderWeightedAseg_w40_cosine.py"
ln -sf "$REPO/scripts/trainers/nnUNetTrainerBorderWeightedAseg_w40_cosineWR.py" "$NNV/nnUNetTrainerBorderWeightedAseg_w40_cosineWR.py"
echo "linked all BorderWeighted* trainers (incl. w40, w40_cosine, w40_cosineWR) -> $NNV"
