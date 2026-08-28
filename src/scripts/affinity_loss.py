"""Torch affinity targets and masked BCE loss for nnUNet affinity trainers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 5:
        return target[:, 0].long()
    if target.dim() == 4:
        return target.long()
    raise ValueError(f"target must be Bx1xZxYxX or BxZxYxX, got shape {tuple(target.shape)}")


def _slices_for_offset(
    shape: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    dz, dy, dx = (int(offset[0]), int(offset[1]), int(offset[2]))
    if dz < 0 or dy < 0 or dx < 0:
        raise ValueError(f"offsets must be non-negative, got {offset}")
    if dz == 0 and dy == 0 and dx == 0:
        raise ValueError("zero offset is not valid")
    z, y, x = shape
    src = (
        slice(0, z - dz if dz else z),
        slice(0, y - dy if dy else y),
        slice(0, x - dx if dx else x),
    )
    dst = (
        slice(dz, z),
        slice(dy, y),
        slice(dx, x),
    )
    return src, dst


def build_torch_affinity_targets(
    target: torch.Tensor,
    offsets: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return same-instance affinity targets and valid mask for local offsets."""
    labels = _normalize_target(target)
    batch, z, y, x = labels.shape
    affinity_target = torch.zeros(
        (batch, len(offsets), z, y, x),
        dtype=torch.float32,
        device=labels.device,
    )
    valid = torch.zeros_like(affinity_target, dtype=torch.bool)
    foreground = labels > 0
    for idx, offset in enumerate(offsets):
        src, dst = _slices_for_offset((z, y, x), offset)
        src_idx = (slice(None), *src)
        dst_idx = (slice(None), *dst)
        edge_valid = foreground[src_idx] & foreground[dst_idx]
        same = edge_valid & (labels[src_idx] == labels[dst_idx])
        valid[(slice(None), idx, *src)] = edge_valid
        affinity_target[(slice(None), idx, *src)] = same.float()
    return affinity_target, valid


def masked_affinity_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCEWithLogits over valid affinity edges only."""
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError(
            f"logits, target, valid must share shape, got {logits.shape}, {target.shape}, {valid.shape}"
        )
    if not torch.any(valid):
        return logits.sum() * 0.0
    weight = torch.ones_like(target)
    if pos_weight != 1.0:
        weight = torch.where(target > 0.5, weight * float(pos_weight), weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
    return loss[valid].mean()
