"""nnUNetTrainerCFSurf — CF-Surf 경계학습 trainer (우승 레버).

기존 nnUNet 손실(deep-supervision DC_CE) 위에 Kervadec boundary loss를 최고해상도에만
추가. alpha를 0→CFSURF_ALPHA(기본 0.3)로 ramp (Kervadec rebalancing).
→ HD95/ASSD/dice/local_dice(8지표 중 4개)를 *직접* 최적화. 추론비용 0.

env:
  CFSURF_ALPHA      boundary 항 최대 가중 (기본 0.3)
  CFSURF_RAMP_FRAC  ramp이 끝나는 학습 진행률 (기본 0.5 = 절반 epoch에 최대 도달)
  CFSURF_FG_CHANNEL softmax foreground 채널 (femur 2-class=1, 기본 1)

⚠️ blanket trainer-scan 충돌 회피: scripts.cf_surf_loss import는 _build_loss 안에서 *lazy*.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _CFSurfDSLoss(nn.Module):
    """nnUNet의 전체 DS 손실(dc_ce)을 그대로 호출 + 최고해상도 출력에 boundary 추가."""

    def __init__(self, ds_loss: nn.Module, boundary: nn.Module, alpha_getter):
        super().__init__()
        self.ds_loss = ds_loss
        self.boundary = boundary
        self.alpha_getter = alpha_getter

    def forward(self, output, target):
        base = self.ds_loss(output, target)
        alpha = self.alpha_getter()
        if alpha and alpha > 0:
            o0 = output[0] if isinstance(output, (list, tuple)) else output
            t0 = target[0] if isinstance(target, (list, tuple)) else target
            base = base + alpha * self.boundary(o0, t0)
        return base


class nnUNetTrainerCFSurf(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # nnUNetTrainer가 inspect.signature(self.__init__)로 init args를 재구성하므로
        # 시그니처를 부모와 *정확히* 일치시켜야 함(*args/**kwargs면 KeyError 'args').
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("CFSURF_EPOCHS", "500"))  # baseline(500)과 공정 A/B
        self.cfsurf_alpha_max = float(os.environ.get("CFSURF_ALPHA", "0.3"))
        self.cfsurf_ramp_frac = float(os.environ.get("CFSURF_RAMP_FRAC", "0.5"))
        self.cfsurf_fg_channel = int(os.environ.get("CFSURF_FG_CHANNEL", "1"))
        self._cfsurf_alpha = 0.0

    def _build_loss(self):
        ds_loss = super()._build_loss()  # nnUNet 표준 DS dc_ce (버전-robust)
        from scripts.cf_surf_loss import BoundaryLoss  # lazy (blanket-scan 회피)
        boundary = BoundaryLoss(fg_channel=self.cfsurf_fg_channel, fg_value=1)
        self.print_to_log_file(
            f"[CF-Surf] boundary loss enabled: alpha_max={self.cfsurf_alpha_max} "
            f"ramp_frac={self.cfsurf_ramp_frac} fg_channel={self.cfsurf_fg_channel}")
        return _CFSurfDSLoss(ds_loss, boundary, lambda: self._cfsurf_alpha)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        ramp_epochs = max(1.0, self.cfsurf_ramp_frac * self.num_epochs)
        self._cfsurf_alpha = self.cfsurf_alpha_max * min(1.0, self.current_epoch / ramp_epochs)
        if self.current_epoch % 25 == 0:
            self.print_to_log_file(f"[CF-Surf] epoch {self.current_epoch} alpha={self._cfsurf_alpha:.4f}")
