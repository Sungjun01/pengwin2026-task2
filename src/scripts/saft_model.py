"""SAFT-GD: Sacrum-Anchored Fracture Transformer model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class SAFTOutput:
    anatomy_logits: torch.Tensor
    fracture_logits: torch.Tensor
    affinity_logits: torch.Tensor
    boundary_logits: torch.Tensor
    seed_logits: torch.Tensor
    severity_logits: torch.Tensor
    embedding: torch.Tensor


class _TinyBackbone(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_swin_unetr(
    in_channels: int,
    feature_channels: int,
    depths: Sequence[int],
    num_heads: Sequence[int],
    window_size: int | Sequence[int],
    use_checkpoint: bool,
) -> nn.Module:
    from monai.networks.nets import SwinUNETR

    return SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=feature_channels,
        feature_size=feature_channels,
        depths=tuple(depths),
        num_heads=tuple(num_heads),
        window_size=window_size,
        use_checkpoint=use_checkpoint,
    )


class SAFTSegmenter(nn.Module):
    """Dense CT model with sacrum-aware input and PENGWIN-specific heads."""

    def __init__(
        self,
        in_channels: int = 6,
        feature_channels: int = 24,
        backbone: str = "swin_unetr",
        anatomy_classes: int = 5,
        affinity_channels: int = 3,
        severity_classes: int = 3,
        swin_depths: Sequence[int] = (2, 2, 2, 2),
        swin_num_heads: Sequence[int] = (3, 6, 12, 24),
        swin_window_size: int | Sequence[int] = 7,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if backbone == "tiny":
            self.backbone = _TinyBackbone(in_channels, feature_channels)
        elif backbone == "swin_unetr":
            self.backbone = _make_swin_unetr(
                in_channels=in_channels,
                feature_channels=feature_channels,
                depths=swin_depths,
                num_heads=swin_num_heads,
                window_size=swin_window_size,
                use_checkpoint=use_checkpoint,
            )
        else:
            raise ValueError(f"unknown SAFT backbone: {backbone}")
        self.anatomy_head = nn.Conv3d(feature_channels, anatomy_classes, kernel_size=1)
        self.fracture_head = nn.Conv3d(feature_channels, 1, kernel_size=1)
        self.affinity_head = nn.Conv3d(feature_channels, affinity_channels, kernel_size=1)
        self.boundary_head = nn.Conv3d(feature_channels, 1, kernel_size=1)
        self.seed_head = nn.Conv3d(feature_channels, 1, kernel_size=1)
        self.severity_head = nn.Conv3d(feature_channels, severity_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SAFTOutput:
        embedding = self.backbone(x)
        return SAFTOutput(
            anatomy_logits=self.anatomy_head(embedding),
            fracture_logits=self.fracture_head(embedding),
            affinity_logits=self.affinity_head(embedding),
            boundary_logits=self.boundary_head(embedding),
            seed_logits=self.seed_head(embedding),
            severity_logits=self.severity_head(embedding),
            embedding=embedding,
        )
