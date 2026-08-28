"""Skeleton Recall Loss (MIC-DKFZ, Kirchhoff/Johannsen et al., ECCV'24) — border-channel 드롭인.

공식 아이디어: 얇은 구조를 skeletonize → 살짝 dilate한 'tube' → 예측이 그 tube를 *recall*(덮게)
강제. clDice처럼 미분가능 soft-skeleton을 매번 돌리지 않고 (OOM 회피), GT에서 skeleton을
precompute(여기선 on-the-fly)하고 recall 항만 → 추론 비용 0, +VRAM 미미.

우리 적용: 얇은 구조 = 골절벽(border seam, 7-class border-core의 클래스 2/4/6). GT border를
skeletonize한 seam 능선을 예측 border-prob이 recall하게 → confident-core(seam을 core로 메움)를
직접 공략. 학습-전용(seg head·추론 그래프 무변경).

  L_skelrec = 1 - sum(p_border * tube) / (sum(tube) + smooth)     # tube 위 border-prob recall
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

BORDER_CLASSES = (2, 4, 6)   # sacrum/LH/RH seam walls


def tubed_skeleton(gt_np: np.ndarray, dilate: int = 1) -> np.ndarray:
    """GT border(2/4/6) union -> skeletonize -> dilate(tube). (Z,Y,X) bool."""
    border = np.isin(gt_np, BORDER_CLASSES)
    if not border.any():
        return np.zeros(gt_np.shape, dtype=bool)
    skel = skeletonize(border)
    if skel.sum() == 0:        # 얇은 seam 벽: skeletonize가 소실 → border 자체가 tube (이미 얇음, dilate 안 함)
        return border
    if dilate > 0:             # 두꺼운 구조의 1-voxel skeleton만 tube화
        skel = ndi.binary_dilation(skel, iterations=dilate)
    return skel


def skeleton_recall_loss(seg_logits: torch.Tensor, label: torch.Tensor,
                         dilate: int = 1, smooth: float = 1.0):
    """Returns scalar = 1 - recall of border-prob over the GT seam skeleton tube."""
    if label.dim() == 5:
        label = label[:, 0]
    label = label.long()
    logits = seg_logits.float()
    with torch.no_grad():
        gt_np = label.cpu().numpy()
        tube_np = np.stack([tubed_skeleton(gt_np[b], dilate) for b in range(gt_np.shape[0])])
    tube = torch.as_tensor(tube_np, device=logits.device, dtype=logits.dtype)   # (B,Z,Y,X)
    if tube.sum() <= 0:
        return logits.sum() * 0.0
    prob = F.softmax(logits, dim=1)
    p_border = sum(prob[:, c] for c in BORDER_CLASSES)                          # (B,Z,Y,X)
    inter = (p_border * tube).sum()
    return 1.0 - (inter + smooth) / (tube.sum() + smooth)
