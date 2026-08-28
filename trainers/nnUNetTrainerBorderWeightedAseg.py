"""F-CIRL trainer: border-weighted DC+CE (config A) + Aseg loss (spec §15, Run 4b).

L_total = L_base + beta * L_Aseg
  L_base = weighted DC+CE (inherited from nnUNetTrainerBorderWeighted, w139 on borders)
  L_Aseg = anatomy-constrained, marginal-masked Aseg loss on a per-voxel log-variance
           head (CentroidConditionedStdHead).

Mechanism (keeps inference cost 0 — the std head is never called at predict time):
  - a forward hook on the decoder's last stage captures the full-res feature;
  - train_step runs the std head on [feature; offset channels] -> log_var;
  - the marginal mask is derived on-the-fly from the GT border classes ({2,4,6} are
    the inter-fragment seam), dilated ~20 mm via GPU max-pool — no instance GT needed.

PELVIC ONLY (D012): needs the 3 offset channels and the LH/RH anatomy split. Femur
(D013, 1 channel, single anatomy) has no offset/side ambiguity, so Aseg is out of scope.

torch.compile is disabled here: compiling the whole network can skip Python forward
hooks on submodules, which would break the feature capture.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.utilities.helpers import dummy_context
from scripts.fcirl_loss import (
    CLASS_TO_ANATOMY,
    CentroidConditionedStdHead,
    aseg_loss_anatomy_constrained,
    marginal_mask_from_label_torch,
)
from scripts.trainers.nnUNetTrainerBorderWeighted import nnUNetTrainerBorderWeighted


class nnUNetTrainerBorderWeightedAseg(nnUNetTrainerBorderWeighted):
    ASEG_BETA = 0.05        # L_total = L_base + beta * L_Aseg (sweep 0.05-0.1)
    ASEG_LAMBDA = 0.5
    MARGIN_RADIUS_MM = 20.0
    OFFSET_CHANNELS = (1, 2, 3)   # input channels carrying the centroid offset

    def _do_i_compile(self):
        return False            # hooks on submodules are unreliable under torch.compile

    @staticmethod
    def _unwrap(net):
        for _ in range(3):
            if hasattr(net, "module"):
                net = net.module           # DDP
            elif hasattr(net, "_orig_mod"):
                net = net._orig_mod        # torch.compile
            else:
                break
        return net

    def _capture_feature_hook(self, module, inputs, output):
        self._last_feature = output

    def initialize(self):
        super().initialize()
        base = self._unwrap(self.network)
        dec = base.decoder
        in_ch = dec.seg_layers[-1].in_channels
        n_cls = self.label_manager.num_segmentation_heads
        self.std_head = CentroidConditionedStdHead(
            in_ch, num_classes=n_cls, offset_channels=len(self.OFFSET_CHANNELS)
        ).to(self.device)
        self._last_feature = None
        dec.stages[-1].register_forward_hook(self._capture_feature_hook)
        self.optimizer.add_param_group({"params": self.std_head.parameters()})

        sp = self.configuration_manager.spacing
        self._margin_iters = max(1, int(round(self.MARGIN_RADIUS_MM / float(np.mean(sp)))))
        c2a = torch.tensor(CLASS_TO_ANATOMY[:n_cls], device=self.device, dtype=torch.long)
        self._class_to_anatomy = c2a
        self.print_to_log_file(
            f"[Aseg] beta={self.ASEG_BETA} lambda={self.ASEG_LAMBDA} "
            f"margin_iters={self._margin_iters} std_head_in={in_ch}")

    def _aseg_term(self, output, data, target):
        seg_logits = output[0] if isinstance(output, (list, tuple)) else output
        full_label = (target[0] if isinstance(target, (list, tuple)) else target)
        if full_label.dim() == 5:
            full_label = full_label[:, 0]
        full_label = full_label.long()
        n_cls = seg_logits.shape[1]

        offset = data[:, self.OFFSET_CHANNELS, ...]
        log_var = self.std_head(self._last_feature, offset).clamp(-10.0, 10.0)
        gt_onehot = F.one_hot(full_label.clamp(0, n_cls - 1), n_cls).permute(0, 4, 1, 2, 3).to(seg_logits.dtype)
        marginal = marginal_mask_from_label_torch(full_label, iters=self._margin_iters)
        anat = self._class_to_anatomy[full_label]
        return aseg_loss_anatomy_constrained(
            seg_logits, log_var, gt_onehot, marginal, anat, lambda_=self.ASEG_LAMBDA)

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
            l = l_base + self.ASEG_BETA * l_aseg

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
