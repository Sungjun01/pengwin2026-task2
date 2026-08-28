"""Dice-guarded Gaussian Skeleton Recall fine-tune wrapper.

This keeps the D016 recipe but makes the recall auxiliary term more conservative:
lower recall weight, Dice guard enabled, and a short low-LR fine-tune cycle.
"""
from __future__ import annotations

import os

import torch

from scripts.trainers.nnUNetTrainerGaussianSkeletonRecall import nnUNetTrainerGaussianSkeletonRecall


class nnUNetTrainerGaussianSkeletonRecallDiceGuard(nnUNetTrainerGaussianSkeletonRecall):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        os.environ.setdefault("PENGWIN_GAUSS_LAMBDA", "0.15")
        os.environ.setdefault("PENGWIN_GAUSS_WDICE", "0.3")
        os.environ.setdefault("PENGWIN_GAUSS_EPOCHS", "20")
        os.environ.setdefault("PENGWIN_GAUSS_LR", "1e-4")
        super().__init__(plans, configuration, fold, dataset_json, device)
