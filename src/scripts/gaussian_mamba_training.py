"""Minimal training utilities for Gaussian-Mamba primitive samples."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gaussian_mamba_model import AnatomyHeadConfig, GaussianMambaOutput, GaussianMambaParser
from scripts.gaussian_mamba_instance_decoder import build_seed_membership_targets


@dataclass(frozen=True)
class GaussianMambaBatch:
    features: torch.Tensor
    centers_zyx: torch.Tensor
    edge_index: torch.Tensor
    edge_same_fragment: torch.Tensor
    edge_weight: torch.Tensor
    targets: dict[str, torch.Tensor]


def load_mamba_sample_npz(path: str | Path, device: str | torch.device = "cpu") -> GaussianMambaBatch:
    data = np.load(path, allow_pickle=True)
    device = torch.device(device)
    if "target_instance_membership" in data and "target_instance_valid" in data:
        instance_membership = data["target_instance_membership"].astype(np.float32)
        instance_valid = data["target_instance_valid"].astype(np.float32)
        instance_fragment_id = data.get("target_instance_fragment_id", np.zeros(instance_valid.shape, dtype=np.int32)).astype(np.int32)
    else:
        membership_targets = build_seed_membership_targets(
            data["target_gt_match_id"],
            data["target_rank_score"],
            max_queries=16,
        )
        instance_membership = membership_targets.membership
        instance_valid = membership_targets.valid_query.astype(np.float32)
        instance_fragment_id = membership_targets.fragment_ids
    targets = {
        "anatomy": torch.as_tensor(data["target_anatomy"], dtype=torch.long, device=device)[None],
        "zone": torch.as_tensor(data["target_zone"], dtype=torch.long, device=device)[None],
        "side": torch.as_tensor(data["target_side"], dtype=torch.long, device=device)[None],
        "structural_role": torch.as_tensor(data["target_structural_role"], dtype=torch.long, device=device)[None],
        "rank_score": torch.as_tensor(data["target_rank_score"], dtype=torch.float32, device=device)[None],
        "gt_match_id": torch.as_tensor(data["target_gt_match_id"], dtype=torch.long, device=device)[None],
        "severity": torch.as_tensor(
            data["target_severity"] if "target_severity" in data else np.zeros_like(data["target_gt_match_id"]),
            dtype=torch.long,
            device=device,
        )[None],
        "importance_weight": torch.as_tensor(
            data["target_importance_weight"] if "target_importance_weight" in data else np.ones_like(data["target_rank_score"]),
            dtype=torch.float32,
            device=device,
        )[None],
        "instance_membership": torch.as_tensor(instance_membership, dtype=torch.float32, device=device)[None],
        "instance_valid": torch.as_tensor(instance_valid, dtype=torch.float32, device=device)[None],
        "instance_fragment_id": torch.as_tensor(instance_fragment_id, dtype=torch.long, device=device)[None],
    }
    return GaussianMambaBatch(
        features=torch.as_tensor(data["features"], dtype=torch.float32, device=device)[None],
        centers_zyx=torch.as_tensor(data["centers_zyx"], dtype=torch.float32, device=device)[None],
        edge_index=torch.as_tensor(data["edge_index"], dtype=torch.long, device=device),
        edge_same_fragment=torch.as_tensor(data["edge_same_fragment"], dtype=torch.float32, device=device)[None],
        edge_weight=torch.as_tensor(
            data["edge_weight"] if "edge_weight" in data else np.ones_like(data["edge_same_fragment"]),
            dtype=torch.float32,
            device=device,
        )[None],
        targets=targets,
    )


def _cross_entropy_token_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    token_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").reshape_as(target)
    if token_weight is None:
        return loss.mean()
    weight = torch.clamp(token_weight, 0.5, 3.0)
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def compute_gaussian_mamba_loss(
    outputs: GaussianMambaOutput,
    batch: GaussianMambaBatch,
    rank_weight: float = 0.5,
    affinity_weight: float = 1.0,
    severity_weight: float = 0.5,
    instance_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    token_weight = batch.targets.get("importance_weight")
    anatomy = _cross_entropy_token_loss(outputs.anatomy_logits, batch.targets["anatomy"], token_weight)
    zone = _cross_entropy_token_loss(outputs.zone_logits, batch.targets["zone"], token_weight)
    side = _cross_entropy_token_loss(outputs.side_logits, batch.targets["side"], token_weight)
    structural_role = _cross_entropy_token_loss(
        outputs.structural_role_logits,
        batch.targets["structural_role"],
        token_weight,
    )
    severity = _cross_entropy_token_loss(outputs.severity_logits, batch.targets["severity"], token_weight)
    rank_loss = F.mse_loss(torch.sigmoid(outputs.foreground_logits), batch.targets["rank_score"], reduction="none")
    if token_weight is None:
        rank = rank_loss.mean()
    else:
        weight = torch.clamp(token_weight, 0.5, 3.0)
        rank = (rank_loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    affinity_pos_weight_value = 1.0
    if outputs.affinity_logits.numel() == 0:
        affinity = outputs.affinity_logits.sum() * 0.0
    else:
        positives = torch.clamp(batch.edge_same_fragment.sum(), min=1.0)
        negatives = torch.clamp(batch.edge_same_fragment.numel() - batch.edge_same_fragment.sum(), min=1.0)
        affinity_pos_weight = torch.clamp(negatives / positives, min=1.0, max=10.0)
        affinity_pos_weight_value = float(affinity_pos_weight.detach().cpu())
        affinity_loss = F.binary_cross_entropy_with_logits(
            outputs.affinity_logits,
            batch.edge_same_fragment,
            pos_weight=affinity_pos_weight,
            reduction="none",
        )
        edge_weight_tensor = torch.clamp(batch.edge_weight, 0.5, 3.0)
        hard_negative = (batch.edge_same_fragment <= 0.0) & (torch.sigmoid(outputs.affinity_logits) >= 0.5)
        edge_weight_tensor = torch.where(hard_negative, torch.clamp(edge_weight_tensor * 1.5, max=3.0), edge_weight_tensor)
        affinity = (affinity_loss * edge_weight_tensor).sum() / torch.clamp(edge_weight_tensor.sum(), min=1.0)
    instance_valid = batch.targets.get("instance_valid")
    instance_membership = batch.targets.get("instance_membership")
    if instance_valid is None or instance_membership is None:
        instance_mask = outputs.instance_mask_logits.sum() * 0.0
        instance_objectness = outputs.instance_objectness_logits.sum() * 0.0
    else:
        query_count = min(outputs.instance_mask_logits.shape[1], instance_membership.shape[1])
        pred_masks = outputs.instance_mask_logits[:, :query_count]
        pred_objectness = outputs.instance_objectness_logits[:, :query_count]
        target_masks = instance_membership[:, :query_count]
        valid = instance_valid[:, :query_count]
        mask_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction="none")
        instance_mask = (mask_loss * valid[:, :, None]).sum() / torch.clamp(valid.sum() * pred_masks.shape[-1], min=1.0)
        instance_objectness = F.binary_cross_entropy_with_logits(pred_objectness, valid, reduction="mean")
    loss = (
        anatomy
        + zone
        + side
        + structural_role
        + severity_weight * severity
        + rank_weight * rank
        + affinity_weight * affinity
        + instance_weight * (instance_mask + instance_objectness)
    )
    components = {
        "anatomy": float(anatomy.detach().cpu()),
        "zone": float(zone.detach().cpu()),
        "side": float(side.detach().cpu()),
        "structural_role": float(structural_role.detach().cpu()),
        "severity": float(severity.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "affinity": float(affinity.detach().cpu()),
        "affinity_pos_weight": affinity_pos_weight_value,
        "instance_mask": float(instance_mask.detach().cpu()),
        "instance_objectness": float(instance_objectness.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    return loss, components


def train_one_epoch(
    model: GaussianMambaParser,
    samples: Iterable[GaussianMambaBatch],
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    last_components: dict[str, float] = {}
    for sample in samples:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sample.features, sample.centers_zyx, sample.edge_index)
        loss, components = compute_gaussian_mamba_loss(outputs, sample)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
        last_components = components
    summary = {"loss": total / max(count, 1), "num_samples": count, **{f"last_{k}": v for k, v in last_components.items()}}
    if checkpoint_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "summary": summary,
            },
            checkpoint_path,
        )
    return summary


def _infer_config_from_sample(sample: GaussianMambaBatch) -> AnatomyHeadConfig:
    return AnatomyHeadConfig(
        anatomy_classes=max(int(sample.targets["anatomy"].max().item()) + 1, 5),
        zone_classes=max(int(sample.targets["zone"].max().item()) + 1, 10),
        side_classes=max(int(sample.targets["side"].max().item()) + 1, 4),
        structural_role_classes=max(int(sample.targets["structural_role"].max().item()) + 1, 6),
        severity_classes=max(int(sample.targets["severity"].max().item()) + 1, 5),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, help="Model-ready Gaussian-Mamba sample .npz files")
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backend", default="gru", choices=("gru", "mamba"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    samples = [load_mamba_sample_npz(path, device=args.device) for path in args.samples]
    first = samples[0]
    model = GaussianMambaParser(
        input_dim=first.features.shape[-1],
        hidden_dim=args.hidden_dim,
        anatomy_config=_infer_config_from_sample(first),
        backend=args.backend,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    summary = {}
    for _ in range(args.epochs):
        summary = train_one_epoch(model, samples, optimizer, checkpoint_path=args.checkpoint)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
