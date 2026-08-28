"""Femur-only topology-locked boundary candidate selection.

The femur path already gets instance counts right on the public femur cases.
This module only edits the femur foreground boundary and rejects candidates that
change the baseline femur ID/component topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.ndimage import label as cc_label


STRUCT26 = np.ones((3, 3, 3), dtype=bool)
FEMUR_ID_RANGE = (151, 200)


@dataclass(frozen=True)
class FemurBoundaryConfig:
    prob_threshold: float = 0.95
    recover_hu_threshold: float = 300.0
    trim_hu_threshold: float = 50.0
    high_hu_reward_threshold: float = 300.0
    low_hu_penalty_threshold: float = 50.0
    max_iters: int = 1
    max_volume_delta_frac: float = 0.05
    min_score_gain: float = 0.0


@dataclass(frozen=True)
class FemurBoundarySelection:
    name: str
    labels: np.ndarray
    score: float
    baseline_score: float
    rejected: list[str]


def _femur_ids(labels: np.ndarray) -> list[int]:
    lo, hi = FEMUR_ID_RANGE
    return sorted(int(x) for x in np.unique(labels) if lo <= int(x) <= hi)


def _femur_mask(labels: np.ndarray) -> np.ndarray:
    lo, hi = FEMUR_ID_RANGE
    return (labels >= lo) & (labels <= hi)


def femur_topology_signature(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return femur IDs with their 26-connected component counts."""
    signature: list[tuple[int, int]] = []
    for iid in _femur_ids(labels):
        _, n_components = cc_label(labels == iid, structure=STRUCT26)
        signature.append((iid, int(n_components)))
    return tuple(signature)


def _project_mask_to_baseline_instances(baseline: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = baseline.copy()
    baseline_femur = _femur_mask(baseline)
    candidate_femur = mask.astype(bool)

    out[baseline_femur & ~candidate_femur] = 0
    added = candidate_femur & ~baseline_femur
    if not added.any() or not baseline_femur.any():
        return out

    _, nearest = distance_transform_edt(~baseline_femur, return_indices=True)
    out[added] = baseline[tuple(axis_indices[added] for axis_indices in nearest)]
    return out


def _recover_shell_mask(seed: np.ndarray, condition: np.ndarray, max_iters: int) -> np.ndarray:
    out = seed.copy()
    for _ in range(max(1, int(max_iters))):
        shell = binary_dilation(out, structure=STRUCT26, iterations=1) & ~out
        recover = shell & condition
        if not recover.any():
            break
        out = out | recover
    return out


def _trim_low_hu_surface(mask: np.ndarray, ct_hu: np.ndarray, cfg: FemurBoundaryConfig) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(1, int(cfg.max_iters))):
        interior = binary_erosion(out, structure=STRUCT26, iterations=1)
        surface = out & ~interior
        trim = surface & (ct_hu < cfg.trim_hu_threshold)
        if not trim.any():
            break
        out = out & ~trim
    return out


def generate_femur_boundary_candidates(
    baseline: np.ndarray,
    ct_hu: np.ndarray,
    femur_probability: np.ndarray,
    cfg: FemurBoundaryConfig | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Generate femur-only boundary candidates projected to baseline instance IDs."""
    cfg = cfg or FemurBoundaryConfig()
    if baseline.shape != ct_hu.shape or baseline.shape != femur_probability.shape:
        raise ValueError("baseline, ct_hu, and femur_probability must have identical shapes")

    base_mask = _femur_mask(baseline)
    prob_mask = _recover_shell_mask(base_mask, femur_probability >= cfg.prob_threshold, cfg.max_iters)
    hu_mask = _recover_shell_mask(base_mask, ct_hu >= cfg.recover_hu_threshold, cfg.max_iters)
    trim_mask = _trim_low_hu_surface(base_mask, ct_hu, cfg)
    trim_hu_mask = _recover_shell_mask(trim_mask, ct_hu >= cfg.recover_hu_threshold, cfg.max_iters)
    trim_prob_mask = _recover_shell_mask(trim_mask, femur_probability >= cfg.prob_threshold, cfg.max_iters)

    return [
        ("prob_recover", _project_mask_to_baseline_instances(baseline, prob_mask)),
        ("hu_recover", _project_mask_to_baseline_instances(baseline, hu_mask)),
        ("trim_low_hu", _project_mask_to_baseline_instances(baseline, trim_mask)),
        ("trim_hu_recover", _project_mask_to_baseline_instances(baseline, trim_hu_mask)),
        ("trim_prob_recover", _project_mask_to_baseline_instances(baseline, trim_prob_mask)),
    ]


def _score(labels: np.ndarray, ct_hu: np.ndarray, cfg: FemurBoundaryConfig) -> float:
    femur = _femur_mask(labels)
    if not femur.any():
        return float("-inf")
    high = np.count_nonzero(femur & (ct_hu >= cfg.high_hu_reward_threshold))
    low = np.count_nonzero(femur & (ct_hu < cfg.low_hu_penalty_threshold))
    return float(high - low)


def _volume_delta_frac(baseline: np.ndarray, candidate: np.ndarray) -> float:
    base_voxels = int(np.count_nonzero(_femur_mask(baseline)))
    if base_voxels == 0:
        return 0.0
    cand_voxels = int(np.count_nonzero(_femur_mask(candidate)))
    return abs(cand_voxels - base_voxels) / float(base_voxels)


def select_femur_boundary_candidate(
    baseline: np.ndarray,
    ct_hu: np.ndarray,
    candidates: Iterable[tuple[str, np.ndarray]],
    cfg: FemurBoundaryConfig | None = None,
) -> FemurBoundarySelection:
    """Pick the best femur candidate while preserving femur IDs and components."""
    cfg = cfg or FemurBoundaryConfig()
    if baseline.shape != ct_hu.shape:
        raise ValueError(f"ct_hu shape {ct_hu.shape} does not match labels shape {baseline.shape}")

    baseline_signature = femur_topology_signature(baseline)
    baseline_score = _score(baseline, ct_hu, cfg)
    best_name = "baseline"
    best_labels = baseline
    best_score = baseline_score
    rejected: list[str] = []

    for name, labels in candidates:
        if femur_topology_signature(labels) != baseline_signature:
            rejected.append(f"{name}:topology")
            continue
        if _volume_delta_frac(baseline, labels) > cfg.max_volume_delta_frac:
            rejected.append(f"{name}:volume")
            continue
        score = _score(labels, ct_hu, cfg)
        if score > best_score + cfg.min_score_gain:
            best_name = name
            best_labels = labels
            best_score = score

    return FemurBoundarySelection(
        name=best_name,
        labels=best_labels.astype(baseline.dtype, copy=False),
        score=best_score,
        baseline_score=baseline_score,
        rejected=rejected,
    )
