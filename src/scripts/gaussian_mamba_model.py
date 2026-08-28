"""Lightweight Gaussian-Mamba parser API.

The production hypothesis is a Mamba state-space parser over Gaussian
primitives. For this research scaffold we keep the same interface and use a
bidirectional GRU fallback when mamba-ssm is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch import nn


@dataclass(frozen=True)
class AnatomyHeadConfig:
    anatomy_classes: int = 4
    zone_classes: int = 10
    side_classes: int = 4
    structural_role_classes: int = 6
    severity_classes: int = 5


@dataclass(frozen=True)
class GaussianMambaOutput:
    anatomy_logits: torch.Tensor
    zone_logits: torch.Tensor
    side_logits: torch.Tensor
    structural_role_logits: torch.Tensor
    severity_logits: torch.Tensor
    foreground_logits: torch.Tensor
    affinity_logits: torch.Tensor
    instance_mask_logits: torch.Tensor
    instance_objectness_logits: torch.Tensor
    embeddings: torch.Tensor

    @property
    def semantic_logits(self) -> torch.Tensor:
        return self.anatomy_logits


class _GRUBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional GRU fallback")
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


def _make_backbone(input_dim: int, hidden_dim: int, backend: str) -> nn.Module:
    if backend == "mamba":
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as exc:
            warnings.warn(
                f"mamba_ssm unavailable ({exc}); using explicit GRU fallback. "
                "Report this run as GRU-backed, not true Mamba.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _GRUBackbone(input_dim, hidden_dim)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Mamba(d_model=hidden_dim),
        )
    if backend == "gru":
        return _GRUBackbone(input_dim, hidden_dim)
    raise ValueError(f"unknown backend: {backend}")


def build_pair_features(
    embeddings: torch.Tensor,
    centers_zyx: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build symmetric pair features for primitive affinity prediction."""
    if edge_index.numel() == 0:
        batch, _, hidden = embeddings.shape
        return embeddings.new_zeros((batch, 0, hidden * 4 + 1))
    i = edge_index[:, 0].long()
    j = edge_index[:, 1].long()
    ei = embeddings[:, i, :]
    ej = embeddings[:, j, :]
    ci = centers_zyx[:, i, :]
    cj = centers_zyx[:, j, :]
    distance = torch.linalg.norm(ci - cj, dim=-1, keepdim=True)
    return torch.cat([ei + ej, torch.abs(ei - ej), ei * ej, (ei + ej) * 0.5, distance], dim=-1)


class GaussianMambaParser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        semantic_classes: int | None = 4,
        anatomy_config: AnatomyHeadConfig | None = None,
        backend: str = "mamba",
        instance_queries: int = 16,
    ) -> None:
        super().__init__()
        if anatomy_config is None:
            anatomy_config = AnatomyHeadConfig(anatomy_classes=int(semantic_classes or 4))
        self.anatomy_config = anatomy_config
        self.instance_queries = int(instance_queries)
        self.backbone = _make_backbone(input_dim, hidden_dim, backend)
        self.anatomy_head = nn.Linear(hidden_dim, anatomy_config.anatomy_classes)
        self.zone_head = nn.Linear(hidden_dim, anatomy_config.zone_classes)
        self.side_head = nn.Linear(hidden_dim, anatomy_config.side_classes)
        self.structural_role_head = nn.Linear(hidden_dim, anatomy_config.structural_role_classes)
        self.severity_head = nn.Linear(hidden_dim, anatomy_config.severity_classes)
        self.foreground_head = nn.Linear(hidden_dim, 1)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.instance_query_embed = nn.Parameter(torch.randn(self.instance_queries, hidden_dim) * 0.02)
        self.instance_token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.instance_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.instance_objectness_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        primitive_features: torch.Tensor,
        centers_zyx: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> GaussianMambaOutput:
        embeddings = self.backbone(primitive_features)
        anatomy_logits = self.anatomy_head(embeddings)
        zone_logits = self.zone_head(embeddings)
        side_logits = self.side_head(embeddings)
        structural_role_logits = self.structural_role_head(embeddings)
        severity_logits = self.severity_head(embeddings)
        foreground_logits = self.foreground_head(embeddings).squeeze(-1)
        if edge_index is None:
            edge_index = torch.empty((0, 2), dtype=torch.long, device=primitive_features.device)
        pair_features = build_pair_features(embeddings, centers_zyx, edge_index.to(primitive_features.device))
        affinity_logits = self.affinity_head(pair_features).squeeze(-1)
        token_proj = self.instance_token_proj(embeddings)
        query_proj = self.instance_query_proj(self.instance_query_embed)[None].expand(embeddings.shape[0], -1, -1)
        instance_mask_logits = torch.einsum("bqh,bnh->bqn", query_proj, token_proj) / max(float(token_proj.shape[-1]) ** 0.5, 1.0)
        instance_objectness_logits = self.instance_objectness_head(query_proj).squeeze(-1)
        return GaussianMambaOutput(
            anatomy_logits=anatomy_logits,
            zone_logits=zone_logits,
            side_logits=side_logits,
            structural_role_logits=structural_role_logits,
            severity_logits=severity_logits,
            foreground_logits=foreground_logits,
            affinity_logits=affinity_logits,
            instance_mask_logits=instance_mask_logits,
            instance_objectness_logits=instance_objectness_logits,
            embeddings=embeddings,
        )
