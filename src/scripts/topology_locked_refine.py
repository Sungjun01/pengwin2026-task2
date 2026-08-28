"""Topology-locked candidate selection for boundary refinement.

The selector is deliberately stricter than a normal post-processor: a candidate
may win only if every original instance ID remains present with the same number
of connected components. This keeps v18's instance/topology behavior locked
while allowing conservative boundary-only edits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import label as cc_label

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class TopologyLockedConfig:
    low_hu_penalty_threshold: float = 50.0
    high_hu_reward_threshold: float = 300.0
    min_score_gain: float = 0.0


@dataclass(frozen=True)
class TopologyLockedSelection:
    name: str
    labels: np.ndarray
    score: float
    baseline_score: float
    rejected: list[str]


def topology_signature(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return nonzero instance IDs with their 26-connected component counts."""
    signature: list[tuple[int, int]] = []
    for iid in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        _, n_components = cc_label(labels == iid, structure=STRUCT26)
        signature.append((iid, int(n_components)))
    return tuple(signature)


def _candidate_score(labels: np.ndarray, ct_hu: np.ndarray, cfg: TopologyLockedConfig) -> float:
    if labels.shape != ct_hu.shape:
        raise ValueError(f"ct_hu shape {ct_hu.shape} does not match labels shape {labels.shape}")
    foreground = labels > 0
    if not foreground.any():
        return float("-inf")
    low_hu = np.count_nonzero(foreground & (ct_hu < cfg.low_hu_penalty_threshold))
    high_hu = np.count_nonzero(foreground & (ct_hu >= cfg.high_hu_reward_threshold))
    return float(high_hu - low_hu)


def select_topology_locked_candidate(
    baseline: np.ndarray,
    ct_hu: np.ndarray,
    candidates: Iterable[tuple[str, np.ndarray]],
    cfg: TopologyLockedConfig | None = None,
) -> TopologyLockedSelection:
    """Pick the best candidate that preserves the baseline topology signature."""
    cfg = cfg or TopologyLockedConfig()
    baseline_signature = topology_signature(baseline)
    baseline_score = _candidate_score(baseline, ct_hu, cfg)
    best_name = "baseline"
    best_labels = baseline
    best_score = baseline_score
    rejected: list[str] = []

    for name, labels in candidates:
        if topology_signature(labels) != baseline_signature:
            rejected.append(name)
            continue
        score = _candidate_score(labels, ct_hu, cfg)
        if score > best_score + cfg.min_score_gain:
            best_name = name
            best_labels = labels
            best_score = score

    return TopologyLockedSelection(
        name=best_name,
        labels=best_labels.astype(baseline.dtype, copy=False),
        score=best_score,
        baseline_score=baseline_score,
        rejected=rejected,
    )
