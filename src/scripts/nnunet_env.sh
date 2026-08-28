#!/usr/bin/env bash
# Source this file (not execute) before running any nnUNet command:
#   source scripts/nnunet_env.sh

export nnUNet_raw=/home/schoaq/taehee/nnUNet_data/raw
export nnUNet_preprocessed=/home/schoaq/taehee/nnUNet_data/preprocessed
export nnUNet_results=/home/schoaq/taehee/nnUNet_data/results

# Consistent GPU enumeration with nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Repo root on PYTHONPATH so nnUNet's trainer-discovery can import custom
# trainers that reference repo modules (e.g. phase_b1_cascade). Without this,
# nnUNetv2_train crashes at class discovery (ModuleNotFoundError: phase_b1_cascade).
export PYTHONPATH="/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee:${PYTHONPATH:-}"

echo "nnUNet_raw          = ${nnUNet_raw}"
echo "nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "nnUNet_results      = ${nnUNet_results}"
