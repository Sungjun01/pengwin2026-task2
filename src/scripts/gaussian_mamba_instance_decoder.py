"""SAM/VLM-inspired instance proposal decoder for Gaussian primitive tokens.

This module is the first step away from raw edge-threshold connected components.
It treats high-confidence primitives as prompt-like seeds and assigns nearby
tokens using sacrum-relative anatomy features, mirroring a SAM-style mask
proposal decoder at primitive-token scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.gaussian_primitives import (
    ANATOMY_SIDE_LABELS,
    GaussianPrimitiveSet,
)


@dataclass(frozen=True)
class SeedDecoderConfig:
    max_seeds: int = 16
    seed_score_threshold: float = 0.35
    seed_nms_distance: float = 0.20
    assignment_threshold: float = 0.40
    same_side_bonus: float = 0.20
    cross_side_penalty: float = 0.35
    same_anatomy_bonus: float = 0.10
    distance_scale: float = 0.35
    min_cluster_size: int = 1


@dataclass(frozen=True)
class SeededInstanceDecode:
    seed_indices: np.ndarray
    membership_scores: np.ndarray
    primitive_labels: np.ndarray


@dataclass(frozen=True)
class SeedMembershipTargets:
    fragment_ids: np.ndarray
    membership: np.ndarray
    valid_query: np.ndarray


def build_seed_membership_targets(
    gt_match_ids: np.ndarray,
    rank_scores: np.ndarray,
    max_queries: int,
) -> SeedMembershipTargets:
    """Build query-mask supervision targets from primitive GT fragment IDs."""
    gt_ids = np.asarray(gt_match_ids, dtype=np.int32)
    scores = np.asarray(rank_scores, dtype=np.float32)
    if gt_ids.shape != scores.shape:
        raise ValueError("gt_match_ids and rank_scores must have the same shape")
    fragment_ids = [int(x) for x in np.unique(gt_ids) if int(x) > 0]
    fragment_ids.sort(key=lambda fid: float(np.max(scores[gt_ids == fid])), reverse=True)
    q = int(max_queries)
    membership = np.zeros((q, len(gt_ids)), dtype=np.float32)
    out_ids = np.zeros((q,), dtype=np.int32)
    valid = np.zeros((q,), dtype=bool)
    for row, fragment_id in enumerate(fragment_ids[:q]):
        out_ids[row] = int(fragment_id)
        valid[row] = True
        membership[row, gt_ids == fragment_id] = 1.0
    return SeedMembershipTargets(fragment_ids=out_ids, membership=membership, valid_query=valid)


def _feature_index(primitive_set: GaussianPrimitiveSet, name: str) -> int | None:
    try:
        return primitive_set.feature_names.index(name)
    except ValueError:
        return None


def _anchor_coordinates(primitive_set: GaussianPrimitiveSet) -> np.ndarray:
    indices = [_feature_index(primitive_set, name) for name in ("sacrum_rel_z", "sacrum_rel_y", "sacrum_rel_x")]
    if all(idx is not None for idx in indices):
        return primitive_set.features[:, [int(idx) for idx in indices]].astype(np.float32)
    centers = primitive_set.centers_zyx.astype(np.float32)
    denom = np.maximum(np.ptp(centers, axis=0, keepdims=True), 1.0)
    return (centers - centers.mean(axis=0, keepdims=True)) / denom


def _side_values(primitive_set: GaussianPrimitiveSet) -> np.ndarray:
    side_idx = _feature_index(primitive_set, "sacrum_side_signed")
    if side_idx is not None:
        return primitive_set.features[:, int(side_idx)].astype(np.float32)
    if primitive_set.anatomy_side_labels is None:
        return np.zeros((len(primitive_set.centers_zyx),), dtype=np.float32)
    labels = primitive_set.anatomy_side_labels.astype(np.int32)
    out = np.zeros(labels.shape, dtype=np.float32)
    out[labels == ANATOMY_SIDE_LABELS["left"]] = -1.0
    out[labels == ANATOMY_SIDE_LABELS["right"]] = 1.0
    return out


def _select_seed_indices(
    coords: np.ndarray,
    scores: np.ndarray,
    config: SeedDecoderConfig,
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    seeds: list[int] = []
    for idx in order:
        idx = int(idx)
        if float(scores[idx]) < float(config.seed_score_threshold):
            continue
        if any(float(np.linalg.norm(coords[idx] - coords[seed])) < float(config.seed_nms_distance) for seed in seeds):
            continue
        seeds.append(idx)
        if len(seeds) >= int(config.max_seeds):
            break
    return np.asarray(seeds, dtype=np.int64)


def _membership_scores(
    primitive_set: GaussianPrimitiveSet,
    seed_indices: np.ndarray,
    foreground_scores: np.ndarray,
    config: SeedDecoderConfig,
) -> np.ndarray:
    coords = _anchor_coordinates(primitive_set)
    sides = _side_values(primitive_set)
    anatomy = primitive_set.anatomy_labels
    scores = np.zeros((len(seed_indices), len(coords)), dtype=np.float32)
    for row, seed_idx in enumerate(seed_indices):
        dist = np.linalg.norm(coords - coords[int(seed_idx)], axis=1)
        score = 1.0 - dist / max(float(config.distance_scale), 1e-6)
        seed_side = float(sides[int(seed_idx)])
        same_side = np.sign(sides) == np.sign(seed_side)
        score += np.where(same_side | (np.abs(sides) < 1e-6) | (abs(seed_side) < 1e-6), config.same_side_bonus, -config.cross_side_penalty)
        if anatomy is not None:
            score += np.where(anatomy == anatomy[int(seed_idx)], config.same_anatomy_bonus, 0.0)
        score += 0.25 * np.asarray(foreground_scores, dtype=np.float32)
        scores[row] = score.astype(np.float32)
    return scores


def decode_seeded_instance_proposals(
    primitive_set: GaussianPrimitiveSet,
    foreground_scores: np.ndarray,
    config: SeedDecoderConfig = SeedDecoderConfig(),
) -> SeededInstanceDecode:
    foreground = np.asarray(foreground_scores, dtype=np.float32)
    if foreground.shape != (len(primitive_set.centers_zyx),):
        raise ValueError("foreground_scores must have shape N")
    coords = _anchor_coordinates(primitive_set)
    seed_indices = _select_seed_indices(coords, foreground, config)
    labels = np.zeros((len(primitive_set.centers_zyx),), dtype=np.int32)
    if len(seed_indices) == 0:
        return SeededInstanceDecode(seed_indices=seed_indices, membership_scores=np.zeros((0, len(labels)), dtype=np.float32), primitive_labels=labels)
    membership = _membership_scores(primitive_set, seed_indices, foreground, config)
    best_seed = np.argmax(membership, axis=0)
    best_score = membership[best_seed, np.arange(membership.shape[1])]
    next_label = 1
    for seed_row in range(len(seed_indices)):
        cluster = (best_seed == seed_row) & (best_score >= float(config.assignment_threshold))
        if int(cluster.sum()) < int(config.min_cluster_size):
            continue
        labels[cluster] = next_label
        next_label += 1
    return SeededInstanceDecode(seed_indices=seed_indices, membership_scores=membership, primitive_labels=labels)
