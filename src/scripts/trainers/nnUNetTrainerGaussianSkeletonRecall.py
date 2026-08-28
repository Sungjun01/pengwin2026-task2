"""가우시안 Soft Skeleton-Recall trainer (드롭인 보조항).

기존 nnUNetTrainerBorderWeightedAseg_w40(D016 레시피) 그대로 +
border-prob 과 가우시안 soft 타겟 G 사이 Soft-BCE/Dice 보조항 (λ 가중).
백본/decode 무손상 — l = l_base + ASEG_BETA·l_aseg + λ·l_gauss.

#1 EDT 최적화: G 는 GPU 가우시안 블러로 매 배치 생성(scripts.gaussian_recall_loss).
#2 λ: GAUSS_LAMBDA 로 main/aux 밸런싱 (env PENGWIN_GAUSS_LAMBDA 로 조정).
#3 fine-tune: GAUSS_EPOCHS(기본 20) + 저LR(env PENGWIN_GAUSS_LR, 기본 1e-4).
   D016 가중치는 nnUNetv2_train -pretrained_weights 로 로드.

env: PENGWIN_GAUSS_SIGMA(1.5) / PENGWIN_GAUSS_LAMBDA(0.3) / PENGWIN_GAUSS_EPOCHS(20) / PENGWIN_GAUSS_LR(1e-4)
"""
from __future__ import annotations
import os
import torch
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import nnUNetTrainerBorderWeightedAseg_w40
from scripts.gaussian_recall_loss import GaussianSkeletonRecallLoss

try:
    from nnunetv2.utilities.helpers import dummy_context
except Exception:
    from contextlib import nullcontext as dummy_context


class nnUNetTrainerGaussianSkeletonRecall(nnUNetTrainerBorderWeightedAseg_w40):
    BORDER_CLASS_INDICES = (2, 4, 6)  # 7-class: sac/LH/RH border

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.GAUSS_SIGMA = float(os.environ.get("PENGWIN_GAUSS_SIGMA", "1.5"))
        self.GAUSS_LAMBDA = float(os.environ.get("PENGWIN_GAUSS_LAMBDA", "0.3"))
        # fine-tune: 짧은 사이클 + 저LR (from-scratch blurring 회피)
        self.num_epochs = int(os.environ.get("PENGWIN_GAUSS_EPOCHS", "20"))
        self.initial_lr = float(os.environ.get("PENGWIN_GAUSS_LR", "1e-4"))
        self._gauss_loss = None

    def initialize(self):
        super().initialize()
        mode = os.environ.get("PENGWIN_GAUSS_MODE", "tversky")  # tversky | atlas | balanced
        alpha = float(os.environ.get("PENGWIN_TVERSKY_ALPHA", "0.3"))
        beta = float(os.environ.get("PENGWIN_TVERSKY_BETA", "0.7"))
        w_dice = float(os.environ.get("PENGWIN_GAUSS_WDICE", "0.0"))
        # atlas-weighted (heatmap 주입): spacing(patch축)·transpose_forward 를 손실에 전달
        atlas_path = os.environ.get("PENGWIN_ATLAS_PATH", "") or None
        atlas_base = float(os.environ.get("PENGWIN_ATLAS_BASE", "0.3"))
        atlas_gamma = float(os.environ.get("PENGWIN_ATLAS_GAMMA", "2.0"))
        spacing = list(self.configuration_manager.spacing)        # patch-축 순서 mm
        tf = list(self.plans_manager.transpose_forward)
        self._gauss_loss = GaussianSkeletonRecallLoss(
            border_class_indices=self.BORDER_CLASS_INDICES, sigma=self.GAUSS_SIGMA,
            mode=mode, alpha=alpha, beta=beta, w_dice=w_dice,
            atlas_path=atlas_path, atlas_base=atlas_base, atlas_gamma=atlas_gamma,
            core_index=1, spacing=spacing, transpose_forward=tf)
        if self._gauss_loss.atlas is not None:
            self._gauss_loss = self._gauss_loss.to(self.device)
        self.print_to_log_file(
            f"[GaussianSkeletonRecall] mode={mode} α={alpha} β={beta} w_dice={w_dice} "
            f"atlas={'ON('+str(atlas_path)+f' base={atlas_base} γ={atlas_gamma})' if atlas_path else 'OFF'} "
            f"spacing={spacing} tf={tf} σ={self.GAUSS_SIGMA} λ={self.GAUSS_LAMBDA} "
            f"epochs={self.num_epochs} lr={self.initial_lr}")

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
            output = self.network(data)                       # hook fills self._last_feature
            l_base = self.loss(output, target)
            l_aseg = self._aseg_term(output, data, target)
            full_logits = output[0] if isinstance(output, (list, tuple)) else output
            full_label = target[0] if isinstance(target, (list, tuple)) else target
            l_gauss = self._gauss_loss(full_logits.float(), full_label)
            l = l_base + self.ASEG_BETA * l_aseg + self.GAUSS_LAMBDA * l_gauss

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
