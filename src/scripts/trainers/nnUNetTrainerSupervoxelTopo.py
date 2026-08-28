"""Supervoxel critical-component topology trainer (arXiv:2501.01022).

Extends the v18 champion (border-weighted w40 DC+CE + F-CIRL Aseg) with ONE train-only
term that directly attacks the dominant merge failure: false-merge bridges (FP components
joining two GT fragments) are penalized per-component (thickness-independent), so a thin
seam bridge gets full gradient. See scripts/supervoxel_topo_loss.py.

    L_total = L_base(w40 DC+CE) + ASEG_BETA*L_Aseg + SV_LAMBDA*(W_MERGE*L_merge + W_SPLIT*L_split)

Train-only: the seg head and inference graph are identical to v18 (the Aseg std-head is also
train-only), so T4 inference cost is unchanged. Unlike a blunt seam penalty, the term fires
ONLY on FP components that bridge >=2 GT fragments, so it should not over-split solid bone
(directly guards the V20 002-break failure mode).

Designed to FINE-TUNE from the champion checkpoint (paper: warm-start from a baseline). For
fold_0 prototyping before any fold_all packaging. Env knobs:
  SV_LAMBDA (0.5)  SV_W_MERGE (1.0)  SV_W_SPLIT (0.25)  SV_MIN_SIZE (2)
  SV_WARMUP (0, epochs before the term turns on)  SV_EPOCHS (override num_epochs)
"""
from __future__ import annotations

import os

import torch

from nnunetv2.utilities.helpers import dummy_context
from scripts.supervoxel_topo_loss import supervoxel_critical_loss
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerSupervoxelTopo(nnUNetTrainerBorderWeightedAseg_w40):
    SV_LAMBDA = float(os.environ.get("SV_LAMBDA", "0.5"))
    W_MERGE = float(os.environ.get("SV_W_MERGE", "1.0"))
    W_SPLIT = float(os.environ.get("SV_W_SPLIT", "0.25"))
    SV_MIN_SIZE = int(os.environ.get("SV_MIN_SIZE", "2"))
    SV_WARMUP = int(os.environ.get("SV_WARMUP", "0"))
    SV_EVERY = int(os.environ.get("SV_EVERY", "1"))   # compute SV term every K steps (CC is CPU-bound)

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch as _torch

        super().__init__(
            plans, configuration, fold, dataset_json,
            device if device is not None else _torch.device("cuda"),
        )
        # warm-start fine-tune from the champion: shorter schedule + lower LR so we
        # refine, not retrain (paper protocol). PolyLR uses these at configure time.
        if os.environ.get("SV_EPOCHS"):
            self.num_epochs = int(os.environ["SV_EPOCHS"])
        if os.environ.get("SV_LR"):
            self.initial_lr = float(os.environ["SV_LR"])

    SNAPSHOT_EVERY = int(os.environ.get("SV_SNAPSHOT_EVERY", "50"))

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            f"[SupervoxelTopo] lambda={self.SV_LAMBDA} w_merge={self.W_MERGE} "
            f"w_split={self.W_SPLIT} min_size={self.SV_MIN_SIZE} "
            f"warmup={self.SV_WARMUP} epochs={self.num_epochs} "
            f"snapshot_every={self.SNAPSHOT_EVERY} lr={self.initial_lr}"
        )

    def on_epoch_end(self):
        # capture BEFORE super() increments self.current_epoch
        ep0 = self.current_epoch
        super().on_epoch_end()  # saves checkpoint_latest/best as usual, then current_epoch += 1
        ep = ep0 + 1
        if (
            self.local_rank == 0
            and self.SNAPSHOT_EVERY > 0
            and ep % self.SNAPSHOT_EVERY == 0
            and ep0 != (self.num_epochs - 1)  # final epoch is saved as checkpoint_final
        ):
            path = os.path.join(self.output_folder, f"checkpoint_ep{ep}.pth")
            self.save_checkpoint(path)
            self.print_to_log_file(f"[SupervoxelTopo] persistent snapshot -> checkpoint_ep{ep}.pth")

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
            output = self.network(data)  # hook fills self._last_feature (Aseg)
            l_base = self.loss(output, target)
            l_aseg = self._aseg_term(output, data, target)
            seg_logits = output[0] if isinstance(output, (list, tuple)) else output
            full_label = self._full_label(target)
            self._sv_step = getattr(self, "_sv_step", 0) + 1
            if self.current_epoch >= self.SV_WARMUP and self._sv_step % self.SV_EVERY == 0:
                l_sv, l_m, l_s = supervoxel_critical_loss(
                    seg_logits,
                    full_label,
                    w_merge=self.W_MERGE,
                    w_split=self.W_SPLIT,
                    min_size=self.SV_MIN_SIZE,
                )
            else:
                l_sv = seg_logits.sum() * 0.0  # keeps graph; zero contribution
                l_m = l_s = 0.0                 # logging-only (avoid float(grad-tensor) warn)
            l = l_base + self.ASEG_BETA * l_aseg + self.SV_LAMBDA * l_sv

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
            "l_merge": float(l_m),
            "l_split": float(l_s),
        }
