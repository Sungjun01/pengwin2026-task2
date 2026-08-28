"""Topology-aware border trainer — separation root-fix (design 2026-06-10).

Extends the F-CIRL Aseg trainer with two train-only topology terms that attack the
confirmed dominant failure (OOF170: 81% separation; report_102: discontinuous
predicted border bridged by 26-conn decode):

    L_total = L_base(w40 DC+CE) + β_aseg·L_Aseg + β_cl·L_clDice + β_seam·L_seamcore

- L_clDice : forces the predicted border to be a *connected wall* (border continuity).
- L_seamcore : on GT seam voxels, punishes "core fills the seam" directly.

Both terms only shape the loss; the seg head and inference graph are unchanged, so
test-time cost is identical to v18 (the std-head is likewise train-only). Terms act
only where the GT has a border, so there is no signal to over-split solid bone
(avoids the V19 decode failure mode where split errors rose).
"""
from __future__ import annotations

import torch

from nnunetv2.utilities.helpers import dummy_context
from scripts.topology_border_loss import cldice_border_loss, seam_core_penalty
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerBorderTopo(nnUNetTrainerBorderWeightedAseg_w40):
    BETA_CLDICE = 0.1          # weight on border-continuity (sweep 0.05-0.3 on fold_0)
    BETA_SEAMCORE = 0.1        # weight on seam-core penalty
    CLDICE_ITERS = 10          # soft-skeleton iterations (>= max seam half-thickness)

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            f"[BorderTopo] beta_cldice={self.BETA_CLDICE} "
            f"beta_seamcore={self.BETA_SEAMCORE} cldice_iters={self.CLDICE_ITERS}")

    @staticmethod
    def _full_label(target):
        full = target[0] if isinstance(target, (list, tuple)) else target
        return full

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            output = self.network(data)                 # hook fills self._last_feature
            l_base = self.loss(output, target)
            l_aseg = self._aseg_term(output, data, target)
            seg_logits = output[0] if isinstance(output, (list, tuple)) else output
            full_label = self._full_label(target)
            l_cldice = cldice_border_loss(seg_logits, full_label, iters=self.CLDICE_ITERS)
            l_seam = seam_core_penalty(seg_logits, full_label)
            l = (l_base
                 + self.ASEG_BETA * l_aseg
                 + self.BETA_CLDICE * l_cldice
                 + self.BETA_SEAMCORE * l_seam)

        params = list(self.network.parameters()) + list(self.std_head.parameters())
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
