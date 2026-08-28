"""w40 + Aseg with CosineAnnealingLR (single smooth cosine decay).

Identical to nnUNetTrainerBorderWeightedAseg_w40 in every respect (border weight 40,
F-CIRL Aseg loss, channel layout, loss composition) except the LR schedule:
PolyLR -> CosineAnnealingLR. T_max = num_epochs (1000), eta_min = 0. This matches the
endpoint of PolyLR (both reach ~0 at the final epoch) but takes a smoother cosine
trajectory — slower decay early, steeper near the end.

Intended as one half of an LR-schedule ablation against the PolyLR baseline (D015
fold_0 GPU8) and the warm-restart variant (cosineWR). Same trainer name pattern so
nnUNet's recursive_find_python_class picks it up after the symlink in
scripts/setup_custom_trainers.sh is extended.
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerBorderWeightedAseg_w40_cosine(nnUNetTrainerBorderWeightedAseg_w40):
    """w40 + Aseg with CosineAnnealingLR(T_max=num_epochs, eta_min=0)."""

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=0.0
        )
        return optimizer, lr_scheduler
