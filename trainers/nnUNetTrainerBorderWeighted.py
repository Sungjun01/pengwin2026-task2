"""1000-epoch nnUNet trainer with CE class weights up-weighting the seam border classes.

This is the LOCKED Task-4 full-training trainer (config A from the A/B/C/D grid,
progress_report_37 / plan §0.1): border weight = 139, seam t=2 (baked into D012/D013
labels). Same loss as nnUNetTrainerBorderWeighted50 but at the nnUNet default 1000 epochs.

Border channel handling is dataset-agnostic via the `if c < n` guard:
    D012 pelvic (7-class): borders = {2, 4, 6} all weighted.
    D013 femur  (3-class): only channel 2 (femur_border) exists -> only it is weighted.

Must be discoverable by nnUNet: symlink/copy into
  <nnunetv2>/training/nnUNetTrainer/variants/  (see scripts/setup_custom_trainers.sh)
"""
import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerBorderWeighted(nnUNetTrainer):
    BORDER_CHANNELS = (2, 4, 6)
    BORDER_WEIGHT = 139.0   # locked (config A); core:border foreground ratio at t=2

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerBorderWeighted is for non-region (argmax) labels"
        n = self.label_manager.num_segmentation_heads
        w = torch.ones(n, device=self.device)
        for c in self.BORDER_CHANNELS:
            if c < n:
                w[c] = self.BORDER_WEIGHT
        self.print_to_log_file(f"[BorderWeighted] CE class weights = {w.tolist()}")

        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': w}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
