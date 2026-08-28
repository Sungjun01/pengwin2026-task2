"""가우시안 Soft Skeleton-Recall Loss (보조항, 드롭인).

설계(사용자): GT 골절면 S → soft 타겟 G(x)=exp(-D²/2σ²) → 예측 border-prob 과
Soft-BCE + Soft-Dice. 약신호(40~80HU) seam 에서 모델이 0/1 hard 페널티 대신
0.3~0.5 연속확률 안전출력 → seam recall↑.

#1 최적화(EDT 병목 회피): EDT/scipy 대신 **GPU 3D 가우시안 블러**로 G 생성.
  얇은 surface 를 N(0,σ²) 로 블러 → 법선거리 D 에서 ∝ exp(-D²/2σ²). peak(=surface)=1
  로 정규화하면 정확히 G. separable conv 3회(z,y,x) = GPU 빠름, 캐싱 불필요,
  augment 된 라벨에서 매 배치 계산하니 augmentation 자동 일치.
#2 λ: total = main + λ·aux. λ 로 main/aux gradient 밸런싱.

border_class_indices: 7-class border-core 의 border 채널들 (sac=2, LH=4, RH=6).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device, dtype, truncate: float = 3.0):
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum(), radius


def gaussian_blur3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """[B,1,D,H,W] 3D separable 가우시안 블러 (GPU)."""
    k, r = _gaussian_kernel1d(sigma, vol.device, vol.dtype)
    out = vol
    for ax in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[ax] = k.numel()
        ker = k.view(shape)
        pad = [0, 0, 0, 0, 0, 0]
        # F.pad order: (W_l,W_r,H_l,H_r,D_l,D_r)
        padcfg = [0, 0, 0, 0, 0, 0]
        if ax == 2: padcfg[4] = padcfg[5] = r
        elif ax == 3: padcfg[2] = padcfg[3] = r
        else: padcfg[0] = padcfg[1] = r
        out = F.conv3d(F.pad(out, padcfg, mode="replicate"), ker)
    return out


def soft_target_from_mask(border_mask: torch.Tensor, sigma: float, eps: float = 1e-6):
    """border surface mask([B,1,...], 0/1) → G=exp(-D²/2σ²) 근사 (peak 정규화)."""
    blurred = gaussian_blur3d(border_mask.float(), sigma)
    # peak = surface 위 값 (블러 자기기여). 정규화: surface voxel 의 블러값으로 나눔.
    # surface 가 있는 배치별 최대값으로 정규화 (peak→1)
    peak = blurred.amax(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
    G = (blurred / peak).clamp(0, 1)
    return G


def soft_bce(pred_prob, target, eps=1e-6):
    p = pred_prob.clamp(eps, 1 - eps)
    return -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()


def soft_dice(pred_prob, target, eps=1e-6):
    num = 2 * (pred_prob * target).sum(dim=(2, 3, 4))
    den = (pred_prob ** 2).sum(dim=(2, 3, 4)) + (target ** 2).sum(dim=(2, 3, 4)) + eps
    return (1 - num / den).mean()


def soft_tversky(pred_prob, target, alpha=0.3, beta=0.7, eps=1e-6):
    """FN-heavy(β>α) → seam 놓침을 헛border보다 무겁게 penalize → recall↑.
    가우시안 균형손실(sharpen=precision)의 회귀를 교정하는 핵심."""
    tp = (pred_prob * target).sum(dim=(2, 3, 4))
    fp = (pred_prob * (1 - target)).sum(dim=(2, 3, 4))
    fn = ((1 - pred_prob) * target).sum(dim=(2, 3, 4))
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).mean()


def weighted_tversky(pred_prob, target, weight, alpha=0.3, beta=0.7, eps=1e-6):
    """voxel-가중 Tversky. weight 높은 곳(atlas-HIGH 예측가능 부위)에서 recall push 집중,
    낮은 곳(atlas-LOW 예측불가)에선 약하게 → 무차별 push가 깬 케이스(063) 회귀 방지."""
    w = weight
    tp = (w * pred_prob * target).sum(dim=(2, 3, 4))
    fp = (w * pred_prob * (1 - target)).sum(dim=(2, 3, 4))
    fn = (w * (1 - pred_prob) * target).sum(dim=(2, 3, 4))
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - t).mean()


class GaussianSkeletonRecallLoss(nn.Module):
    """보조 가우시안 soft-recall 손실. forward(logits, target_label) → scalar.

    mode='tversky'(기본): FN-heavy soft_tversky(α<β) → recall↑. balanced(soft_bce+soft_dice)가
      border 를 sharpen(precision)해서 RQ −0.105 회귀시킨 것을 교정. w_dice>0 이면 약한 dice 항 병행.
    mode='balanced': 옛 soft_bce+soft_dice (회귀 재현용, 비교 전용)."""
    def __init__(self, border_class_indices=(2, 4, 6), sigma=1.5,
                 mode="tversky", alpha=0.3, beta=0.7, w_tversky=1.0,
                 w_dice=0.0, w_bce=1.0,
                 atlas_path=None, atlas_base=0.3, atlas_gamma=2.0, core_index=1,
                 spacing=None, transpose_forward=(0, 1, 2)):
        super().__init__()
        self.border_idx = tuple(border_class_indices)
        self.sigma = sigma
        self.mode = mode
        self.alpha, self.beta = alpha, beta
        self.w_tversky, self.w_dice, self.w_bce = w_tversky, w_dice, w_bce
        # --- atlas-weighted (heatmap을 손실에 주입) ---
        self.atlas_base, self.atlas_gamma, self.core_index = atlas_base, atlas_gamma, core_index
        self.spacing = tuple(spacing) if spacing is not None else None  # patch-축 순서 [s0,s1,s2] mm
        self.tf = tuple(int(t) for t in transpose_forward)              # plans transpose_forward
        self.atlas = None
        if atlas_path:
            import numpy as _np
            d = _np.load(atlas_path)
            a = d["atlas"].astype("float32")
            self.register_buffer("_atlas", torch.from_numpy(a))
            self.atlas_R = float(d["R"]); self.atlas_BIN = float(d["BIN"]); self.atlas_N = int(d["N"])
            self.atlas = True
        # atlas 모드면 mode 와 무관하게 weighted_tversky 사용 (mode=='atlas')

    def _atlas_weight_map(self, target_label):
        """target sac_core 중심으로 population atlas 를 patch 프레임에 배치 → W=base+γ·atlas."""
        B = target_label.shape[0]
        P = tuple(target_label.shape[2:])  # (P0,P1,P2) patch 축
        dev = target_label.device
        W = torch.full((B, 1) + P, float(self.atlas_base), device=dev, dtype=torch.float32)
        if self.atlas is None or self.spacing is None:
            return W
        atlas = self._atlas.to(dev)
        Nn = self.atlas_N; R = self.atlas_R; BN = self.atlas_BIN
        # atlas 축 k(0=z,1=y,2=x) 를 담은 patch 축: pa[k] = tf.index(k)
        pa = [self.tf.index(0), self.tf.index(1), self.tf.index(2)]
        for b in range(B):
            core = (target_label[b, 0] == self.core_index)
            if core.sum() < 10:
                continue
            cc = core.nonzero(as_tuple=False).float().mean(0)  # patch coords (c0,c1,c2)
            idx_k = []; valid_k = []
            for k in range(3):                 # atlas axis k
                ax = pa[k]                      # patch axis holding it
                coord = torch.arange(P[ax], device=dev, dtype=torch.float32)
                off = (coord - cc[ax]) * float(self.spacing[ax])
                ii = ((off + R) / BN).long()
                vv = (off >= -R) & (off < R)
                shape = [1, 1, 1]; shape[ax] = P[ax]
                idx_k.append(ii.clamp(0, Nn - 1).view(shape).expand(P))
                valid_k.append(vv.view(shape).expand(P))
            av = atlas[idx_k[0], idx_k[1], idx_k[2]]       # [P0,P1,P2], atlas[z,y,x]
            valid = valid_k[0] & valid_k[1] & valid_k[2]
            av = av * valid.float()
            W[b, 0] = self.atlas_base + self.atlas_gamma * av
        return W

    def forward(self, logits: torch.Tensor, target_label: torch.Tensor):
        # logits [B,C,...]; target_label [B,1,...] or [B,...]
        if target_label.dim() == logits.dim() - 1:
            target_label = target_label.unsqueeze(1)
        prob = torch.softmax(logits, dim=1)
        # 예측: voxel 이 *어느 border 든* 일 확률 = border 채널 합
        border_prob = prob[:, self.border_idx, ...].sum(dim=1, keepdim=True).clamp(0, 1)
        # GT border surface mask
        bm = torch.zeros_like(target_label, dtype=torch.float32)
        for idx in self.border_idx:
            bm = bm + (target_label == idx).float()
        bm = bm.clamp(0, 1)
        if bm.sum() < 1:
            return border_prob.sum() * 0.0  # border 없으면 0 (미분 연결 유지)
        G = soft_target_from_mask(bm, self.sigma)
        if self.mode == "atlas":
            W = self._atlas_weight_map(target_label)
            l = self.w_tversky * weighted_tversky(border_prob, G, W, self.alpha, self.beta)
            if self.w_dice > 0:
                l = l + self.w_dice * soft_dice(border_prob, G)
            return l
        if self.mode == "tversky":
            l = self.w_tversky * soft_tversky(border_prob, G, self.alpha, self.beta)
            if self.w_dice > 0:
                l = l + self.w_dice * soft_dice(border_prob, G)
            return l
        # balanced (옛 회귀 재현용)
        return self.w_bce * soft_bce(border_prob, G) + self.w_dice * soft_dice(border_prob, G)
