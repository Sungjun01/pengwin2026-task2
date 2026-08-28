"""GT-supervised statistical ranking for Gaussian primitives."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from scripts.gaussian_primitives import ANATOMY_LABELS, GaussianPrimitiveSet


COMPONENT_NAMES = ("purity", "coverage", "fit", "shape", "hu", "seam", "clutter")


@dataclass(frozen=True)
class PrimitiveGtMatch:
    primitive_index: int
    gt_id: int
    support_count: int
    majority_count: int
    purity: float
    coverage: float
    mean_mahalanobis: float
    fit: float
    shape_score: float
    hu_likelihood: float
    seam_score: float
    clutter_penalty: float


@dataclass(frozen=True)
class PrimitiveRankWeights:
    purity: float = 0.25
    coverage: float = 0.20
    fit: float = 0.15
    shape: float = 0.15
    hu: float = 0.15
    seam: float = 0.10
    clutter: float = 0.25
    rare_fragment_bonus: float = 0.15
    fragment_representative_bonus: float = 0.30


@dataclass(frozen=True)
class PrimitiveRankResult:
    rank_scores: np.ndarray
    gt_match_ids: np.ndarray
    rank_components: np.ndarray
    component_names: tuple[str, ...]
    matches: tuple[PrimitiveGtMatch, ...]


def ellipsoid_support_indices(
    center_zyx: np.ndarray,
    covariance: np.ndarray,
    volume_shape: tuple[int, int, int],
    mahalanobis_radius: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center_zyx, dtype=np.float32)
    cov = np.asarray(covariance, dtype=np.float32)
    eigvals = np.linalg.eigvalsh(cov)
    max_sigma = float(np.sqrt(max(float(np.max(eigvals)), 1e-3)))
    radius = int(np.ceil(mahalanobis_radius * max_sigma))
    lo = np.maximum(np.floor(center).astype(np.int32) - radius, 0)
    hi = np.minimum(np.ceil(center).astype(np.int32) + radius + 1, np.asarray(volume_shape, dtype=np.int32))
    grids = np.meshgrid(
        np.arange(lo[0], hi[0]),
        np.arange(lo[1], hi[1]),
        np.arange(lo[2], hi[2]),
        indexing="ij",
    )
    coords = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.float32)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    inv_cov = np.linalg.pinv(cov + np.eye(3, dtype=np.float32) * 1e-3)
    diff = coords - center[None, :]
    mahal = np.einsum("ni,ij,nj->n", diff, inv_cov, diff).astype(np.float32)
    keep = mahal <= float(mahalanobis_radius) ** 2
    return coords[keep].astype(np.int32), mahal[keep]


def _shape_score(covariance: np.ndarray, anatomy_label: int | None) -> float:
    eigvals = np.sort(np.linalg.eigvalsh(covariance).astype(np.float32))
    eigvals = np.maximum(eigvals, 1e-3)
    anisotropy = float(eigvals[-1] / eigvals[0])
    if anatomy_label == ANATOMY_LABELS["femur"]:
        return float(np.clip((anisotropy - 1.0) / 5.0, 0.0, 1.0))
    if anatomy_label == ANATOMY_LABELS["sacrum"]:
        return float(np.clip(1.0 - abs(anisotropy - 2.0) / 4.0, 0.0, 1.0))
    if anatomy_label == ANATOMY_LABELS["pelvic_ring"]:
        return float(np.clip(1.0 - abs(anisotropy - 3.0) / 5.0, 0.0, 1.0))
    return float(np.clip(1.0 / anisotropy, 0.0, 1.0))


def _hu_likelihood(local_hu: np.ndarray, anatomy_label: int | None) -> float:
    if len(local_hu) == 0:
        return 0.0
    hu_mean = float(np.mean(local_hu))
    expected = 350.0
    sigma = 250.0
    if anatomy_label == ANATOMY_LABELS["pelvic_ring"]:
        expected = 260.0
    elif anatomy_label == ANATOMY_LABELS["sacrum"]:
        expected = 220.0
    return float(np.exp(-((hu_mean - expected) ** 2) / (2.0 * sigma**2)))


def compute_primitive_gt_match(
    primitive_set: GaussianPrimitiveSet,
    primitive_index: int,
    gt_labels: np.ndarray,
    ct_hu: np.ndarray,
    mahalanobis_radius: float = 2.5,
) -> PrimitiveGtMatch:
    gt = np.asarray(gt_labels)
    ct = np.asarray(ct_hu, dtype=np.float32)
    coords, mahal = ellipsoid_support_indices(
        primitive_set.centers_zyx[primitive_index],
        primitive_set.covariances[primitive_index],
        gt.shape,
        mahalanobis_radius=mahalanobis_radius,
    )
    if len(coords) == 0:
        return PrimitiveGtMatch(primitive_index, 0, 0, 0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 1.0)

    support_labels = gt[tuple(coords.T)]
    positive = support_labels[support_labels > 0]
    center_idx = np.rint(primitive_set.centers_zyx[primitive_index]).astype(np.int64)
    center_idx = np.maximum(0, np.minimum(center_idx, np.asarray(gt.shape, dtype=np.int64) - 1))
    center_gt_id = int(gt[tuple(center_idx)])
    if center_gt_id > 0 and np.any(positive == center_gt_id):
        gt_id = center_gt_id
        majority_count = int(np.sum(positive == center_gt_id))
    elif len(positive) == 0:
        gt_id = 0
        majority_count = 0
    else:
        ids, counts = np.unique(positive, return_counts=True)
        best = int(np.argmax(counts))
        gt_id = int(ids[best])
        majority_count = int(counts[best])

    support_count = int(len(coords))
    purity = float(majority_count / max(len(positive), 1)) if majority_count > 0 else 0.0
    coverage = 0.0
    if gt_id > 0:
        coverage = float(majority_count / max(int(np.sum(gt == gt_id)), 1))
    mean_mahalanobis = float(np.mean(mahal)) if len(mahal) else float("inf")
    fit = float(np.exp(-mean_mahalanobis / 3.0)) if np.isfinite(mean_mahalanobis) else 0.0
    anatomy_label = None
    if primitive_set.anatomy_labels is not None:
        anatomy_label = int(primitive_set.anatomy_labels[primitive_index])
    local_hu = ct[tuple(coords.T)]
    shape_score = _shape_score(primitive_set.covariances[primitive_index], anatomy_label)
    hu_likelihood = _hu_likelihood(local_hu, anatomy_label)
    low_hu_ratio = float(np.mean(local_hu <= 80.0)) if len(local_hu) else 1.0
    seam_score = float(np.clip(low_hu_ratio, 0.0, 1.0))
    background_fraction = float(np.mean(support_labels == 0))
    small_support_penalty = 1.0 if support_count < 3 else 0.0
    clutter_penalty = float(np.clip(0.7 * background_fraction + 0.3 * small_support_penalty, 0.0, 1.0))
    return PrimitiveGtMatch(
        primitive_index=primitive_index,
        gt_id=gt_id,
        support_count=support_count,
        majority_count=majority_count,
        purity=purity,
        coverage=coverage,
        mean_mahalanobis=mean_mahalanobis,
        fit=fit,
        shape_score=shape_score,
        hu_likelihood=hu_likelihood,
        seam_score=seam_score,
        clutter_penalty=clutter_penalty,
    )


def _score_match(match: PrimitiveGtMatch, weights: PrimitiveRankWeights) -> tuple[float, list[float]]:
    components = [
        match.purity,
        match.coverage,
        match.fit,
        match.shape_score,
        match.hu_likelihood,
        match.seam_score,
        match.clutter_penalty,
    ]
    score = (
        weights.purity * match.purity
        + weights.coverage * match.coverage
        + weights.fit * match.fit
        + weights.shape * match.shape_score
        + weights.hu * match.hu_likelihood
        + weights.seam * match.seam_score
        - weights.clutter * match.clutter_penalty
    )
    return float(score), components


def rank_gaussian_primitives(
    primitive_set: GaussianPrimitiveSet,
    gt_labels: np.ndarray,
    ct_hu: np.ndarray,
    weights: PrimitiveRankWeights | None = None,
) -> PrimitiveRankResult:
    if weights is None:
        weights = PrimitiveRankWeights()
    matches = []
    scores = []
    components = []
    gt_ids = []
    gt_sizes = {int(x): int(np.sum(gt_labels == x)) for x in np.unique(gt_labels) if int(x) > 0}
    max_gt_size = max(gt_sizes.values(), default=1)
    for idx in range(len(primitive_set.centers_zyx)):
        match = compute_primitive_gt_match(primitive_set, idx, gt_labels, ct_hu)
        score, comp = _score_match(match, weights)
        if match.gt_id > 0:
            rarity = 1.0 - (gt_sizes.get(match.gt_id, max_gt_size) / max_gt_size)
            score += weights.rare_fragment_bonus * float(np.clip(rarity, 0.0, 1.0))
        if primitive_set.importance_weights is not None:
            importance = float(np.clip(primitive_set.importance_weights[idx], 0.5, 3.0))
            score += 0.02 * (importance - 1.0)
        matches.append(match)
        scores.append(score)
        components.append(comp)
        gt_ids.append(match.gt_id)
    scores_arr = np.asarray(scores, dtype=np.float32)
    gt_ids_arr = np.asarray(gt_ids, dtype=np.int32)
    for gt_id in sorted(set(int(x) for x in gt_ids_arr if int(x) > 0)):
        fragment_indices = np.where(gt_ids_arr == gt_id)[0]
        if len(fragment_indices) == 0:
            continue
        best_local = fragment_indices[int(np.argmax(scores_arr[fragment_indices]))]
        scores_arr[best_local] += weights.fragment_representative_bonus
    return PrimitiveRankResult(
        rank_scores=scores_arr,
        gt_match_ids=gt_ids_arr,
        rank_components=np.asarray(components, dtype=np.float32),
        component_names=COMPONENT_NAMES,
        matches=tuple(matches),
    )


def attach_rank_result_to_primitives(
    primitive_set: GaussianPrimitiveSet,
    rank_result: PrimitiveRankResult,
) -> GaussianPrimitiveSet:
    """Return a primitive set with GT-supervised rank fields attached."""
    return replace(
        primitive_set,
        rank_scores=rank_result.rank_scores,
        gt_match_ids=rank_result.gt_match_ids,
        rank_components=rank_result.rank_components,
        rank_component_names=rank_result.component_names,
    )


def topk_fragment_recall(
    gt_match_ids: np.ndarray,
    rank_scores: np.ndarray,
    required_gt_ids: list[int] | tuple[int, ...] | np.ndarray,
    k: int,
) -> float:
    required = {int(x) for x in required_gt_ids if int(x) > 0}
    if not required:
        return 1.0
    positive_order = [
        int(gt_match_ids[idx])
        for idx in np.argsort(np.asarray(rank_scores))[::-1]
        if int(gt_match_ids[idx]) > 0
    ]
    recovered = set(positive_order[: max(int(k), 0)])
    return float(len(required & recovered) / len(required))
