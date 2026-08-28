"""Decision-layer decode utilities for Gaussian-Mamba primitive outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.gaussian_primitives import (
    GaussianPrimitiveSet,
    decode_primitive_affinities,
    splat_primitives_to_voxels,
)
from scripts.surface_refine import SurfaceRefineConfig, refine_instance_surfaces


@dataclass(frozen=True)
class DecisionConfig:
    foreground_threshold: float = 0.5
    micro_keep_threshold: float = 0.35
    attractive_threshold: float = 0.5
    repulsive_threshold: float = 0.5
    temperature: float = 1.0
    target_range: tuple[int, int] = (1, 255)
    preserve_important_representative: bool = False
    representative_importance_threshold: float = 2.0


@dataclass(frozen=True)
class BoundaryRefineConfig:
    enabled: bool = False
    trim_hu_threshold: float = 50.0
    recover_hu_threshold: float = 300.0
    max_iters: int = 1
    min_keep_voxels: int = 8


@dataclass(frozen=True)
class DecodedGaussianMambaCase:
    keep_mask: np.ndarray
    edge_index: np.ndarray
    attractive_scores: np.ndarray
    repulsive_scores: np.ndarray
    primitive_labels: np.ndarray
    instance_mask: np.ndarray


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _build_keep_mask(
    primitive_set: GaussianPrimitiveSet,
    foreground_logits: np.ndarray,
    config: DecisionConfig,
) -> np.ndarray:
    foreground = _sigmoid(foreground_logits)
    keep = foreground >= float(config.foreground_threshold)
    if primitive_set.importance_weights is not None:
        important = primitive_set.importance_weights >= 2.0
        keep |= important & (foreground >= float(config.micro_keep_threshold))
    if primitive_set.severity_labels is not None:
        micro = primitive_set.severity_labels == 1
        keep |= micro & (foreground >= float(config.micro_keep_threshold))
    if config.preserve_important_representative and primitive_set.importance_weights is not None:
        important = np.where(primitive_set.importance_weights >= float(config.representative_importance_threshold))[0]
        if len(important) and not np.any(keep[important]):
            best = important[int(np.argmax(foreground[important]))]
            keep[best] = True
    return keep.astype(bool)


def decode_gaussian_mamba_case(
    primitive_set: GaussianPrimitiveSet,
    foreground_logits: np.ndarray,
    edge_index: np.ndarray,
    affinity_logits: np.ndarray,
    foreground_mask: np.ndarray,
    config: DecisionConfig = DecisionConfig(),
    ct_hu: np.ndarray | None = None,
    boundary_refine: BoundaryRefineConfig | None = None,
) -> DecodedGaussianMambaCase:
    keep_mask = _build_keep_mask(primitive_set, foreground_logits, config)
    edges = np.asarray(edge_index, dtype=np.int32)
    edge_keep = np.asarray([keep_mask[i] and keep_mask[j] for i, j in edges], dtype=bool)
    kept_edges = edges[edge_keep]
    affinity = _sigmoid(np.asarray(affinity_logits, dtype=np.float32)[edge_keep] / max(float(config.temperature), 1e-3))
    uncertainty = 1.0 - np.abs(affinity - 0.5) * 2.0
    repulsive = np.maximum(1.0 - affinity, uncertainty)

    primitive_labels = np.zeros((len(primitive_set.centers_zyx),), dtype=np.int32)
    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices):
        local_index = {int(old): idx for idx, old in enumerate(kept_indices)}
        local_edges = np.asarray(
            [[local_index[int(i)], local_index[int(j)]] for i, j in kept_edges],
            dtype=np.int32,
        )
        local_labels = decode_primitive_affinities(
            local_edges,
            affinity,
            repulsive,
            n_primitives=len(kept_indices),
            attractive_threshold=config.attractive_threshold,
            repulsive_threshold=config.repulsive_threshold,
        )
        primitive_labels[kept_indices] = local_labels

    instance_mask = splat_primitives_to_voxels(
        primitive_set,
        primitive_labels,
        np.asarray(foreground_mask, dtype=bool),
        target_range=config.target_range,
    )
    if boundary_refine is not None and boundary_refine.enabled:
        if ct_hu is None:
            raise ValueError("ct_hu is required when boundary_refine is enabled")
        instance_mask = refine_instance_surfaces(
            instance_mask,
            np.asarray(ct_hu, dtype=np.float32),
            cfg=SurfaceRefineConfig(
                trim_hu_threshold=boundary_refine.trim_hu_threshold,
                recover_hu_threshold=boundary_refine.recover_hu_threshold,
                max_iters=boundary_refine.max_iters,
                min_keep_voxels=boundary_refine.min_keep_voxels,
            ),
        )
    return DecodedGaussianMambaCase(
        keep_mask=keep_mask,
        edge_index=kept_edges,
        attractive_scores=affinity.astype(np.float32),
        repulsive_scores=repulsive.astype(np.float32),
        primitive_labels=primitive_labels,
        instance_mask=instance_mask,
    )
