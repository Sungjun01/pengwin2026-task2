"""Losses for standalone Voxel Mamba segmentation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.affinity_loss import build_torch_affinity_targets, masked_affinity_bce_loss
from scripts.voxel_mamba_model import VoxelMambaOutput

DEFAULT_AFFINITY_OFFSETS: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _normalize_label_shape(label: torch.Tensor) -> torch.Tensor:
    if label.dim() == 5:
        return label[:, 0].long()
    if label.dim() == 4:
        return label.long()
    raise ValueError(f"label must be BxZxYxX or Bx1xZxYxX, got {tuple(label.shape)}")


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: int = 5,
    eps: float = 1e-6,
    include_background: bool = True,
) -> torch.Tensor:
    """Soft Dice loss over `classes` channels.

    When `include_background` is False the background channel (index 0) is
    dropped from the mean. This matters for sparse-foreground volumes: with the
    background included, a degenerate all-background prediction scores
    `(1 + 0*(C-1)) / C` Dice -> loss ~0.8, a low-loss basin the optimizer falls
    into (the PENGWIN SAFT collapse). Excluding background makes an
    all-background prediction score Dice ~0 -> loss ~1.0, so it is no longer a
    minimum.
    """
    target = _normalize_label_shape(target)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.clamp_min(0), num_classes=classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    if not include_background:
        dice = dice[1:]
    return 1.0 - dice.mean()


def boundary_target_from_fragments(fragment_label: torch.Tensor) -> torch.Tensor:
    labels = _normalize_label_shape(fragment_label)
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    for dim in (1, 2, 3):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[dim] = slice(0, -1)
        right[dim] = slice(1, None)
        a = labels[tuple(left)]
        b = labels[tuple(right)]
        edge = (a > 0) & (b > 0) & (a != b)
        boundary[tuple(left)] |= edge
        boundary[tuple(right)] |= edge
    return boundary[:, None].float()


def compute_voxel_mamba_loss(
    outputs: VoxelMambaOutput,
    anatomy_label: torch.Tensor,
    fragment_label: torch.Tensor,
    anatomy_weight: float = 1.0,
    affinity_weight: float = 0.5,
    boundary_weight: float = 0.25,
    affinity_offsets: tuple[tuple[int, int, int], ...] = DEFAULT_AFFINITY_OFFSETS,
) -> tuple[torch.Tensor, dict[str, float]]:
    anatomy_label = _normalize_label_shape(anatomy_label).to(outputs.anatomy_logits.device)
    fragment_label = _normalize_label_shape(fragment_label).to(outputs.anatomy_logits.device)
    anatomy_ce = F.cross_entropy(outputs.anatomy_logits, anatomy_label)
    anatomy_dice = soft_dice_loss(outputs.anatomy_logits, anatomy_label, classes=outputs.anatomy_logits.shape[1])
    affinity_target, affinity_valid = build_torch_affinity_targets(fragment_label, affinity_offsets)
    affinity = masked_affinity_bce_loss(outputs.affinity_logits, affinity_target, affinity_valid)
    boundary_target = boundary_target_from_fragments(fragment_label).to(outputs.boundary_logits.device)
    boundary = F.binary_cross_entropy_with_logits(outputs.boundary_logits, boundary_target)
    loss = anatomy_weight * (anatomy_ce + anatomy_dice) + affinity_weight * affinity + boundary_weight * boundary
    return loss, {
        "anatomy_ce": float(anatomy_ce.detach().cpu()),
        "anatomy_dice": float(anatomy_dice.detach().cpu()),
        "affinity": float(affinity.detach().cpu()),
        "boundary": float(boundary.detach().cpu()),
    }
