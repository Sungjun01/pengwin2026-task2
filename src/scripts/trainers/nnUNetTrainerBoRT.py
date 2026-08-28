"""BoRT trainer (Skea-Topo, IJCAI'24) — skeleton-aware narrow-seam weighting on the pelvic
border-core. Extends the v18 champion (w40 DC+CE + F-CIRL Aseg) with one train-only term:

    L_total = L_base(w40 DC+CE) + ASEG_BETA*L_Aseg + BORT_LAMBDA*(L_skeaw + BORT_TOPO*L_topo)

L_skeaw upweights THIN inter-fragment seams (where the model merges) inversely to local seam
thickness; L_topo keeps the seam skeleton predicted as a wall. Train-only: seg head + inference
graph identical to v18 (T4-safe). On-the-fly from GT (no instance-label dataset needed).

Designed to FINE-TUNE the champion for a fast fold_0 probe (warm-start, same as Supervoxel
probe) OR train from scratch. Env knobs:
  BORT_LAMBDA(0.5) BORT_W0(10) BORT_TOPO(0.1) BORT_WMAX_MM(8) BORT_WARMUP(0)
  BORT_SNAPSHOT_EVERY(50) BORT_EPOCHS(override num_epochs) BORT_LR(override lr)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from nnunetv2.utilities.helpers import dummy_context
from scripts.bort_loss import bort_loss
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerBoRT(nnUNetTrainerBorderWeightedAseg_w40):
    BORT_LAMBDA = float(os.environ.get("BORT_LAMBDA", "0.5"))
    BORT_W0 = float(os.environ.get("BORT_W0", "10.0"))
    BORT_TOPO = float(os.environ.get("BORT_TOPO", "0.1"))
    BORT_WMAX_MM = float(os.environ.get("BORT_WMAX_MM", "8.0"))
    BORT_WARMUP = int(os.environ.get("BORT_WARMUP", "0"))
    SNAPSHOT_EVERY = int(os.environ.get("BORT_SNAPSHOT_EVERY", "50"))

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(
            plans, configuration, fold, dataset_json,
            device if device is not None else torch.device("cuda"),
        )
        if os.environ.get("BORT_EPOCHS"):
            self.num_epochs = int(os.environ["BORT_EPOCHS"])
        if os.environ.get("BORT_LR"):
            self.initial_lr = float(os.environ["BORT_LR"])

    def initialize(self):
        super().initialize()
        self._mean_sp = float(np.mean(self.configuration_manager.spacing))
        self.print_to_log_file(
            f"[BoRT] lambda={self.BORT_LAMBDA} w0={self.BORT_W0} topo={self.BORT_TOPO} "
            f"wmax_mm={self.BORT_WMAX_MM} warmup={self.BORT_WARMUP} "
            f"snapshot_every={self.SNAPSHOT_EVERY} epochs={self.num_epochs} "
            f"lr={self.initial_lr} mean_spacing={self._mean_sp:.2f}mm"
        )

    def on_epoch_end(self):
        ep0 = self.current_epoch
        super().on_epoch_end()
        ep = ep0 + 1
        if (
            self.local_rank == 0
            and self.SNAPSHOT_EVERY > 0
            and ep % self.SNAPSHOT_EVERY == 0
            and ep0 != (self.num_epochs - 1)
        ):
            path = os.path.join(self.output_folder, f"checkpoint_ep{ep}.pth")
            self.save_checkpoint(path)
            self.print_to_log_file(f"[BoRT] persistent snapshot -> checkpoint_ep{ep}.pth")

    @staticmethod
    def _full_label(target):
        return target[0] if isinstance(target, (list, tuple)) else target

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with ctx:
            output = self.network(data)
            l_base = self.loss(output, target)
            l_aseg = self._aseg_term(output, data, target)
            seg_logits = output[0] if isinstance(output, (list, tuple)) else output
            full_label = self._full_label(target)
            if self.current_epoch >= self.BORT_WARMUP:
                l_bort, l_skeaw, l_topo = bort_loss(
                    seg_logits, full_label, w0=self.BORT_W0, lambda_topo=self.BORT_TOPO,
                    wmax_mm=self.BORT_WMAX_MM, spacing_mm=self._mean_sp,
                )
            else:
                l_bort = seg_logits.sum() * 0.0
                l_skeaw = l_topo = 0.0
            l = l_base + self.ASEG_BETA * l_aseg + self.BORT_LAMBDA * l_bort

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
        return {
            "loss": l.detach().cpu().numpy(),
            "l_skeaw": float(l_skeaw),
            "l_topo": float(l_topo),
        }
