"""Full 3D CT-aware Voxel Mamba segmentation model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VoxelMambaOutput:
    anatomy_logits: torch.Tensor
    affinity_logits: torch.Tensor
    boundary_logits: torch.Tensor
    embedding: torch.Tensor


def _import_mamba_cls() -> Type[nn.Module]:
    from mamba_ssm import Mamba  # type: ignore

    return Mamba


def require_mamba_cls() -> Type[nn.Module]:
    try:
        return _import_mamba_cls()
    except Exception as exc:
        raise RuntimeError(
            "true mamba_ssm.Mamba is required for VoxelMambaUNet; "
            "GRU/RNN fallback is intentionally disabled for dense CT training"
        ) from exc


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VoxelMambaBlock(nn.Module):
    """Apply true Mamba over Z, Y, and X axial token sequences."""

    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        mamba_cls = require_mamba_cls()
        self.norm = nn.LayerNorm(channels)
        self.blocks = nn.ModuleList([mamba_cls(d_model=channels) for _ in range(int(depth))])
        self.out = nn.Conv3d(channels, channels, kernel_size=1)

    def _scan(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        b, c, z, y, w = x.shape
        if axis == 0:
            tokens = x.permute(0, 3, 4, 2, 1).reshape(b * y * w, z, c)
            shape = (b, y, w, z, c)
            inverse = lambda t: t.reshape(shape).permute(0, 4, 3, 1, 2)
        elif axis == 1:
            tokens = x.permute(0, 2, 4, 3, 1).reshape(b * z * w, y, c)
            shape = (b, z, w, y, c)
            inverse = lambda t: t.reshape(shape).permute(0, 4, 1, 3, 2)
        elif axis == 2:
            tokens = x.permute(0, 2, 3, 4, 1).reshape(b * z * y, w, c)
            shape = (b, z, y, w, c)
            inverse = lambda t: t.reshape(shape).permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        tokens = self.norm(tokens)
        for block in self.blocks:
            tokens = tokens + block(tokens)
        return inverse(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scanned = (self._scan(x, 0) + self._scan(x, 1) + self._scan(x, 2)) / 3.0
        return x + self.out(scanned)


class VoxelMambaUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        anatomy_classes: int = 5,
        affinity_channels: int = 3,
        mamba_depth: int = 1,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvNormAct(in_channels, c)
        self.enc2 = ConvNormAct(c, c * 2, stride=2)
        self.enc3 = ConvNormAct(c * 2, c * 4, stride=2)
        self.mid_mamba = VoxelMambaBlock(c * 4, depth=mamba_depth)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvNormAct(c * 4, c * 2)
        self.mamba2 = VoxelMambaBlock(c * 2, depth=mamba_depth)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvNormAct(c * 2, c)
        self.anatomy_head = nn.Conv3d(c, anatomy_classes, kernel_size=1)
        self.affinity_head = nn.Conv3d(c, affinity_channels, kernel_size=1)
        self.boundary_head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> VoxelMambaOutput:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.mid_mamba(self.enc3(e2))
        d2 = self.up2(e3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.mamba2(self.dec2(torch.cat([d2, e2], dim=1)))
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
        embedding = self.dec1(torch.cat([d1, e1], dim=1))
        return VoxelMambaOutput(
            anatomy_logits=self.anatomy_head(embedding),
            affinity_logits=self.affinity_head(embedding),
            boundary_logits=self.boundary_head(embedding),
            embedding=embedding,
        )
