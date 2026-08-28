"""50-epoch nnUNet trainer with CE class weights up-weighting the seam border classes.

Border-core overfit gate escalation step 1: the seam border is ~0.71% of foreground
(139:1), so the standard DC+CE trainer leaves border pseudo-dice at 0.0 (never wins
argmax). This trainer up-weights the border channels in the CE term to test whether
the thin seam becomes learnable.

7-class border-core label order:
    0 bg, 1 sacrum_core, 2 sacrum_border, 3 LH_core, 4 LH_border, 5 RH_core, 6 RH_border
-> border channels = {2, 4, 6}.

Mirrors nnUNetTrainer._build_loss exactly, only adding ce_kwargs={'weight': ...}.
Must be discoverable by nnUNet: symlink/copy into
  <nnunetv2>/training/nnUNetTrainer/variants/  (see scripts/setup_custom_trainers.sh)
"""
import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerBorderWeighted50(nnUNetTrainer):
    BORDER_CHANNELS = (2, 4, 6)
    BORDER_WEIGHT = 139.0   # inverse foreground core:border ratio; drop to ~12 if unstable

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerBorderWeighted50 is for non-region (argmax) labels"
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
