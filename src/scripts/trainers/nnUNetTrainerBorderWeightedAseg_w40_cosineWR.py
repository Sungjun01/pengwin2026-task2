"""w40 + Aseg with CosineAnnealingWarmRestarts (SGDR, 4 equal cycles).

Identical to nnUNetTrainerBorderWeightedAseg_w40 except the LR schedule:
PolyLR -> CosineAnnealingWarmRestarts(T_0=250, T_mult=1, eta_min=1e-6). Over 1000
epochs the LR follows four cosine cycles (epochs 0-249, 250-499, 500-749, 750-999),
each starting at initial_lr (1e-2) and falling to eta_min (1e-6) before snapping back
to initial_lr at the cycle boundary.

The warm-restart "shake" is a deliberate perturbation: it can pop the model out of a
local minimum (mode-collapse equivalent class boundary) and may discover a flatter
basin. We expect to take cycle-end checkpoints (ep 249, 499, 749, 999) and treat them
as an ensemble candidate set.

NOTE on nnUNet's per-epoch step: the base trainer calls
    self.lr_scheduler.step(self.current_epoch)
at the start of every training epoch. CosineAnnealingWarmRestarts.step(epoch=...) is
the documented call form for SGDR (it sets T_cur from the absolute epoch), so this
plays nicely with nnUNet's loop and survives checkpoint resume.
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerBorderWeightedAseg_w40_cosineWR(nnUNetTrainerBorderWeightedAseg_w40):
    """w40 + Aseg with CosineAnnealingWarmRestarts(T_0=250, T_mult=1, eta_min=1e-6)."""

    T_0 = 250         # first cycle length (epochs)
    T_MULT = 1        # equal cycle lengths -> restarts at 250, 500, 750
    ETA_MIN = 1e-6    # cosine floor (above 0 to keep gradient signal in restart region)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_MULT,
            eta_min=self.ETA_MIN,
        )
        return optimizer, lr_scheduler
