"""Femur affinity prototype trainer.

Research trainer for the affinity + Mutex Watershed pivot. The nnUNet semantic
head remains the normal foreground/background output. An auxiliary 1x1x1
affinity head is attached to the final decoder feature map and trained to
predict same-fragment local affinities for configured offsets.

This trainer is intended for fold_0 prototyping before any fold_all packaging.
Inference support must use the same offsets and a separate decoder gate.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from scripts.affinity_loss import build_torch_affinity_targets, masked_affinity_bce_loss

try:
    from torch import autocast
except ImportError:  # pragma: no cover
    from contextlib import nullcontext as autocast


class nnUNetTrainerFemurAffinity(nnUNetTrainer):
    """nnUNet semantic trainer plus auxiliary local-affinity BCE."""

    AFFINITY_OFFSETS = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
    )

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("AFFINITY_EPOCHS", "500"))
        self.affinity_lambda = float(os.environ.get("AFFINITY_LAMBDA", "0.2"))
        self.affinity_pos_weight = float(os.environ.get("AFFINITY_POS_WEIGHT", "1.0"))
        self._last_feature = None

    @staticmethod
    def _unwrap(net):
        while True:
            if hasattr(net, "module"):
                net = net.module
            elif hasattr(net, "_orig_mod"):
                net = net._orig_mod
            else:
                return net

    def _capture_feature_hook(self, module, inputs, output):
        self._last_feature = output

    def initialize(self):
        super().initialize()
        base = self._unwrap(self.network)
        dec = base.decoder
        in_ch = dec.seg_layers[-1].in_channels
        self.affinity_head = nn.Conv3d(in_ch, len(self.AFFINITY_OFFSETS), kernel_size=1).to(self.device)
        self._last_feature = None
        dec.stages[-1].register_forward_hook(self._capture_feature_hook)
        self.optimizer.add_param_group({"params": self.affinity_head.parameters()})
        self.print_to_log_file(
            "[FemurAffinity] "
            f"lambda={self.affinity_lambda} pos_weight={self.affinity_pos_weight} "
            f"offsets={self.AFFINITY_OFFSETS} head_in={in_ch}"
        )

    def _affinity_term(self, target):
        if self._last_feature is None:
            raise RuntimeError("affinity feature hook did not run")
        full_label = target[0] if isinstance(target, (list, tuple)) else target
        if full_label.dim() == 5:
            spatial_shape = tuple(int(x) for x in full_label.shape[2:])
        else:
            spatial_shape = tuple(int(x) for x in full_label.shape[1:])
        affinity_logits = self.affinity_head(self._last_feature)
        if tuple(affinity_logits.shape[2:]) != spatial_shape:
            affinity_logits = F.interpolate(
                affinity_logits,
                size=spatial_shape,
                mode="trilinear",
                align_corners=False,
            )
        affinity_target, valid = build_torch_affinity_targets(full_label, self.AFFINITY_OFFSETS)
        return masked_affinity_bce_loss(
            affinity_logits.float(),
            affinity_target,
            valid,
            pos_weight=self.affinity_pos_weight,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else torch.no_grad()
        if self.device.type != "cuda":
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx:
            output = self.network(data)
            l_base = self.loss(output, target)
            l_aff = self._affinity_term(target)
            loss = l_base + self.affinity_lambda * l_aff

        params = list(self.network.parameters()) + list(self.affinity_head.parameters())
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_affinity": l_aff.detach().cpu().numpy(),
        }
