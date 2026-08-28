"""Adaptive capacity planning for Gaussian-Mamba primitive parsing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityConfig:
    base_primitives: int = 128
    base_topk: int = 32
    max_primitives: int = 384
    max_topk: int = 128
    base_min_primitives_per_fragment: int = 4


@dataclass(frozen=True)
class CaseComplexityEstimate:
    estimated_fragments: int
    micro_fragment_count: int = 0
    complex_zone_count: int = 0
    mean_importance: float = 1.0


@dataclass(frozen=True)
class CapacityPlan:
    num_primitives: int
    topk: int
    min_primitives_per_fragment: int
    complexity_score: float


def plan_adaptive_capacity(
    estimate: CaseComplexityEstimate,
    config: CapacityConfig = CapacityConfig(),
) -> CapacityPlan:
    fragments = max(int(estimate.estimated_fragments), 0)
    micro = max(int(estimate.micro_fragment_count), 0)
    zones = max(int(estimate.complex_zone_count), 0)
    importance = float(np.clip(estimate.mean_importance, 0.5, 3.0))

    overload = max(fragments - 8, 0)
    zone_overload = max(zones - 1, 0)
    complexity = overload * 3.0 + micro * 4.0 + zone_overload * 6.0 + max(importance - 1.0, 0.0) * 16.0
    primitives = int(np.ceil(config.base_primitives + complexity))
    topk = int(np.ceil(config.base_topk + overload * 1.2 + micro * 1.5 + zone_overload * 2.0))
    min_quota = int(np.clip(config.base_min_primitives_per_fragment + micro // 12, 2, 8))

    primitives = int(np.clip(primitives, config.base_primitives, config.max_primitives))
    topk = int(np.clip(topk, config.base_topk, min(config.max_topk, primitives)))
    return CapacityPlan(
        num_primitives=primitives,
        topk=topk,
        min_primitives_per_fragment=min_quota,
        complexity_score=float(complexity),
    )
