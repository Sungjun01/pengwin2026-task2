"""CF-Surf 경계손실 — 우승 레버(8지표 중 HD95/ASSD/dice/local_dice 4개 직접 공략).

Kervadec boundary loss(2019): L_bd = mean( softmax_fg · sdf_gt ),
  sdf_gt = GT의 부호거리(내부<0, 외부>0, 표면=0).
  최소화하면 예측 경계가 GT 표면으로 당겨짐 → HD95/ASSD를 *직접* 줄임
  (Dice/CE는 표면에서 거의 평평해 경계 오차를 못 봄).

train-only(추론비용 0), 미분가능(pred에 대해), GT-side만 필요(sdf는 detached weight).
nnUNet과 결합: DC_CE + alpha·boundary, alpha를 0.01→0.3 ramp.

self-contained: torch/numpy/scipy만 import (eez191 blanket trainer-scan 충돌 회피).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as _edt, find_objects


def _sdf_from_label(label_zyx: np.ndarray, fg_value: int = 1, margin: int = 24) -> np.ndarray:
    """단일 label volume(int) → 부호거리장(float32). 내부<0, 외부>0.
    fg bbox+margin 만 EDT 후 되돌려 가속(표면 ±margin 밖은 클립값이라 정보 없음).
    crop 밖은 +margin(큰 양수=표면서 멀음) 채움 → 거기 pred는 이미 0이라 grad 무의미."""
    fg = (label_zyx == fg_value)
    if not fg.any() or fg.all():
        return np.zeros(label_zyx.shape, dtype=np.float32)
    sl = find_objects(fg.astype(np.int8))[0]
    sl = tuple(slice(max(0, s.start - margin), min(d, s.stop + margin))
               for s, d in zip(sl, label_zyx.shape))
    sub = fg[sl]
    sub_sdf = (_edt(~sub) - _edt(sub)).astype(np.float32)
    out = np.full(label_zyx.shape, float(margin), dtype=np.float32)
    out[sl] = sub_sdf
    return out


def batch_sdf(target: torch.Tensor, fg_value: int = 1) -> torch.Tensor:
    """target (B,1,Z,Y,X) 또는 (B,Z,Y,X) int → sdf (B,1,Z,Y,X) float, 같은 device.
    표면 ±band만 의미 있게 쓰도록 클립(±20 voxel)해 outlier 영향 제한."""
    if target.dim() == target.new_zeros(0).dim():
        pass
    t = target
    if t.dim() == 5:
        t = t[:, 0]
    elif t.dim() != 4:
        raise ValueError(f"target dim {target.dim()} 예상밖")
    arr = t.detach().cpu().numpy().astype(np.int16)
    out = np.empty((arr.shape[0],) + arr.shape[1:], dtype=np.float32)
    for b in range(arr.shape[0]):
        s = _sdf_from_label(arr[b], fg_value)
        out[b] = np.clip(s, -20.0, 20.0)
    return torch.from_numpy(out).unsqueeze(1).to(target.device)


class BoundaryLoss(nn.Module):
    """Kervadec boundary loss. net_output: (B,C,...) logits. target: (B,1,...) int labels.
    fg_channel = softmax foreground 채널 index (femur 2-class면 1)."""

    def __init__(self, fg_channel: int = 1, fg_value: int = 1, downsample: int = 2):
        super().__init__()
        self.fg_channel = fg_channel
        self.fg_value = fg_value
        self.downsample = int(downsample)   # EDT 가속: 표면을 1/d 해상도에서 평가

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(net_output, dim=1)
        pc = probs[:, self.fg_channel:self.fg_channel + 1]   # (B,1,Z,Y,X)
        t = target
        d = self.downsample
        if d > 1:
            pc = torch.nn.functional.avg_pool3d(pc, d)        # 확률 평균
            tt = t.float()
            if tt.dim() == 4:
                tt = tt[:, None]
            tt = torch.nn.functional.max_pool3d(tt, d)        # 라벨 보존(nearest≈max for 0/1)
            t = tt
        with torch.no_grad():
            sdf = batch_sdf(t, self.fg_value)                 # detached weight
        return (pc * sdf).mean()


class DC_CE_Boundary(nn.Module):
    """기존 nnUNet 손실(dc_ce) + alpha·boundary. alpha는 trainer가 epoch마다 set."""

    def __init__(self, dc_ce_loss: nn.Module, fg_channel: int = 1, fg_value: int = 1,
                 alpha: float = 0.0):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.boundary = BoundaryLoss(fg_channel, fg_value)
        self.alpha = float(alpha)

    def forward(self, net_output, target):
        base = self.dc_ce(net_output, target)
        if self.alpha <= 0:
            return base
        return base + self.alpha * self.boundary(net_output, target)
