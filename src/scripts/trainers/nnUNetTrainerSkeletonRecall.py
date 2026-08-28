"""Skeleton Recall trainer (MIC-DKFZ, ECCV'24) — border-channel drop-in on the v18 champion.

    L_total = L_base(w40 DC+CE) + ASEG_BETA*L_Aseg + SKEL_LAMBDA*L_skelrecall

L_skelrecall pushes predicted border-prob to recall the GT seam skeleton tube (connectivity of
the thin separating wall). Train-only: seg head + inference graph identical to v18 (T4-safe).
Warm-start from the champion (-pretrained_weights) for a fast fold_0 gate, like the Supervoxel/
BoRT probes. Env: SKEL_LAMBDA(0.5) SKEL_DILATE(1) SKEL_EVERY(4, skeletonize is CPU)
SKEL_WARMUP(0) SKEL_SNAPSHOT_EVERY(20) SKEL_EPOCHS SKEL_LR.
"""
from __future__ import annotations

import os

import torch

from nnunetv2.utilities.helpers import dummy_context
from scripts.skeleton_recall_loss import skeleton_recall_loss
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerSkeletonRecall(nnUNetTrainerBorderWeightedAseg_w40):
    SKEL_LAMBDA = float(os.environ.get("SKEL_LAMBDA", "0.5"))
    SKEL_DILATE = int(os.environ.get("SKEL_DILATE", "1"))
    SKEL_EVERY = int(os.environ.get("SKEL_EVERY", "4"))
    SKEL_WARMUP = int(os.environ.get("SKEL_WARMUP", "0"))
    SNAPSHOT_EVERY = int(os.environ.get("SKEL_SNAPSHOT_EVERY", "20"))

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json,
                         device if device is not None else torch.device("cuda"))
        if os.environ.get("SKEL_EPOCHS"):
            self.num_epochs = int(os.environ["SKEL_EPOCHS"])
        if os.environ.get("SKEL_LR"):
            self.initial_lr = float(os.environ["SKEL_LR"])

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            f"[SkeletonRecall] lambda={self.SKEL_LAMBDA} dilate={self.SKEL_DILATE} "
            f"every={self.SKEL_EVERY} warmup={self.SKEL_WARMUP} epochs={self.num_epochs} "
            f"snapshot={self.SNAPSHOT_EVERY} lr={self.initial_lr}")

    def on_epoch_end(self):
        ep0 = self.current_epoch
        super().on_epoch_end()
        ep = ep0 + 1
        if (self.local_rank == 0 and self.SNAPSHOT_EVERY > 0
                and ep % self.SNAPSHOT_EVERY == 0 and ep0 != (self.num_epochs - 1)):
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_ep{ep}.pth"))
            self.print_to_log_file(f"[SkeletonRecall] snapshot -> checkpoint_ep{ep}.pth")

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
        ctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == "cuda" else dummy_context())
        with ctx:
            output = self.network(data)
            l_base = self.loss(output, target)
            l_aseg = self._aseg_term(output, data, target)
            seg_logits = output[0] if isinstance(output, (list, tuple)) else output
            full_label = self._full_label(target)
            self._skel_step = getattr(self, "_skel_step", 0) + 1
            if self.current_epoch >= self.SKEL_WARMUP and self._skel_step % self.SKEL_EVERY == 0:
                l_skel = skeleton_recall_loss(seg_logits, full_label, dilate=self.SKEL_DILATE)
            else:
                l_skel = seg_logits.sum() * 0.0
            l = l_base + self.ASEG_BETA * l_aseg + self.SKEL_LAMBDA * l_skel

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
        return {"loss": l.detach().cpu().numpy(), "l_skel": float(l_skel.detach())}
