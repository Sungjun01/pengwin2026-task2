"""SAFT-GD losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.saft_model import SAFTOutput
from scripts.voxel_mamba_losses import (
    DEFAULT_AFFINITY_OFFSETS,
    boundary_target_from_fragments,
    soft_dice_loss,
)
from scripts.affinity_loss import build_torch_affinity_targets, masked_affinity_bce_loss


def severity_target_from_fragments(fragment_label: torch.Tensor) -> torch.Tensor:
    """Voxel severity target: background=0, micro=0, minor=1, major=2 for foreground."""
    labels = fragment_label.long()
    out = torch.zeros_like(labels, dtype=torch.long)
    for batch_idx in range(labels.shape[0]):
        ids = sorted(int(x) for x in set(torch.unique(labels[batch_idx]).detach().cpu().tolist()) - {0})
        for frag_id in ids:
            count = int((labels[batch_idx] == frag_id).sum().item())
            if count < 64:
                sev = 0
            elif count < 512:
                sev = 1
            else:
                sev = 2
            out[batch_idx][labels[batch_idx] == frag_id] = sev
    return out


def compute_saft_loss(
    outputs: SAFTOutput,
    anatomy_label: torch.Tensor,
    fragment_label: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs.anatomy_logits.device
    anatomy_label = anatomy_label.long().to(device)
    fragment_label = fragment_label.long().to(device)
    anatomy_ce = F.cross_entropy(outputs.anatomy_logits, anatomy_label)
    # Exclude background from the Dice mean: with background included a pure
    # all-background prediction is a low-loss basin (loss ~0.8) the optimizer
    # collapses into on sparse-foreground PENGWIN patches.
    anatomy_dice = soft_dice_loss(
        outputs.anatomy_logits,
        anatomy_label,
        classes=outputs.anatomy_logits.shape[1],
        include_background=False,
    )
    foreground = (fragment_label > 0).float()[:, None]
    fracture = F.binary_cross_entropy_with_logits(outputs.fracture_logits, foreground)
    boundary_target = boundary_target_from_fragments(fragment_label)
    boundary = F.binary_cross_entropy_with_logits(outputs.boundary_logits, boundary_target)
    affinity_target, affinity_valid = build_torch_affinity_targets(fragment_label, DEFAULT_AFFINITY_OFFSETS)
    affinity = masked_affinity_bce_loss(outputs.affinity_logits, affinity_target, affinity_valid)
    seed = F.binary_cross_entropy_with_logits(outputs.seed_logits, foreground)
    severity_target = severity_target_from_fragments(fragment_label).to(device)
    severity = F.cross_entropy(outputs.severity_logits, severity_target)
    loss = anatomy_ce + anatomy_dice + 0.5 * fracture + 0.5 * affinity + 0.25 * boundary + 0.25 * seed + 0.25 * severity
    return loss, {
        "anatomy_ce": float(anatomy_ce.detach().cpu()),
        "anatomy_dice": float(anatomy_dice.detach().cpu()),
        "fracture": float(fracture.detach().cpu()),
        "affinity": float(affinity.detach().cpu()),
        "boundary": float(boundary.detach().cpu()),
        "seed": float(seed.detach().cpu()),
        "severity": float(severity.detach().cpu()),
    }
