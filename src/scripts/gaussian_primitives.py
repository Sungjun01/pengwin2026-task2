"""Gaussian primitive utilities for the PENGWIN Gaussian-Mamba prototype.

This module is intentionally lightweight and offline-first. It turns a CT +
foreground mask into anisotropic ellipsoid tokens, labels those tokens from GT
instances, and provides a primitive graph decode/splat path for research tests.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ANATOMY_LABELS: dict[str, int] = {
    "unknown": 0,
    "sacrum": 1,
    "pelvic_ring": 2,
    "acetabulum_hip": 3,
    "femur": 4,
}

ANATOMY_ZONE_LABELS: dict[str, int] = {
    "unknown": 0,
    "sacral_body": 1,
    "sacral_ala": 2,
    "ilium": 3,
    "pubis": 4,
    "ischium": 5,
    "acetabulum": 6,
    "femoral_head_neck": 7,
    "trochanter": 8,
    "shaft": 9,
}

ANATOMY_SIDE_LABELS: dict[str, int] = {
    "unknown": 0,
    "left": 1,
    "right": 2,
    "midline": 3,
}

ANATOMY_STRUCTURAL_ROLES: dict[str, int] = {
    "unknown": 0,
    "ring_bridge": 1,
    "joint_surface": 2,
    "long_bone_axis": 3,
    "small_fragment": 4,
    "clutter_candidate": 5,
}


@dataclass(frozen=True)
class GaussianPrimitiveSet:
    centers_zyx: np.ndarray
    covariances: np.ndarray
    features: np.ndarray
    support_counts: np.ndarray
    feature_names: tuple[str, ...]
    fragment_ids: np.ndarray | None = None
    anatomy_labels: np.ndarray | None = None
    anatomy_zone_labels: np.ndarray | None = None
    anatomy_side_labels: np.ndarray | None = None
    structural_role_labels: np.ndarray | None = None
    rank_scores: np.ndarray | None = None
    gt_match_ids: np.ndarray | None = None
    rank_components: np.ndarray | None = None
    rank_component_names: tuple[str, ...] | None = None
    severity_labels: np.ndarray | None = None
    importance_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(self.centers_zyx.shape[0])
        if self.centers_zyx.shape != (n, 3):
            raise ValueError(f"centers_zyx must have shape Nx3, got {self.centers_zyx.shape}")
        if self.covariances.shape != (n, 3, 3):
            raise ValueError(f"covariances must have shape Nx3x3, got {self.covariances.shape}")
        if self.features.shape[0] != n:
            raise ValueError("features first dimension must match centers")
        if self.support_counts.shape != (n,):
            raise ValueError("support_counts must have shape N")
        if self.fragment_ids is not None and self.fragment_ids.shape != (n,):
            raise ValueError("fragment_ids must have shape N")
        for name in (
            "anatomy_labels",
            "anatomy_zone_labels",
            "anatomy_side_labels",
            "structural_role_labels",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (n,):
                raise ValueError(f"{name} must have shape N")
        if self.rank_scores is not None and self.rank_scores.shape != (n,):
            raise ValueError("rank_scores must have shape N")
        if self.gt_match_ids is not None and self.gt_match_ids.shape != (n,):
            raise ValueError("gt_match_ids must have shape N")
        if self.rank_components is not None and self.rank_components.shape[0] != n:
            raise ValueError("rank_components first dimension must match centers")
        if self.rank_components is not None and self.rank_component_names is None:
            raise ValueError("rank_component_names are required when rank_components are present")
        if self.severity_labels is not None and self.severity_labels.shape != (n,):
            raise ValueError("severity_labels must have shape N")
        if self.importance_weights is not None and self.importance_weights.shape != (n,):
            raise ValueError("importance_weights must have shape N")


@dataclass(frozen=True)
class PrimitiveAffinityEdge:
    i: int
    j: int
    same_fragment: bool
    distance: float


@dataclass(frozen=True)
class SacrumAnchorFrame:
    centroid_zyx: np.ndarray
    spacing_zyx: np.ndarray
    norm_mm: float


def save_primitive_set(path: str | Path, primitive_set: GaussianPrimitiveSet) -> None:
    fragment_ids = (
        primitive_set.fragment_ids
        if primitive_set.fragment_ids is not None
        else np.full(primitive_set.centers_zyx.shape[0], -1, dtype=np.int32)
    )
    n = primitive_set.centers_zyx.shape[0]

    def optional_labels(values: np.ndarray | None) -> np.ndarray:
        return values if values is not None else np.full(n, -1, dtype=np.int32)

    def optional_float(values: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        return values if values is not None else np.full(shape, np.nan, dtype=np.float32)

    np.savez_compressed(
        path,
        centers_zyx=primitive_set.centers_zyx.astype(np.float32),
        covariances=primitive_set.covariances.astype(np.float32),
        features=primitive_set.features.astype(np.float32),
        support_counts=primitive_set.support_counts.astype(np.int32),
        feature_names=np.asarray(primitive_set.feature_names, dtype=object),
        fragment_ids=fragment_ids.astype(np.int32),
        anatomy_labels=optional_labels(primitive_set.anatomy_labels).astype(np.int32),
        anatomy_zone_labels=optional_labels(primitive_set.anatomy_zone_labels).astype(np.int32),
        anatomy_side_labels=optional_labels(primitive_set.anatomy_side_labels).astype(np.int32),
        structural_role_labels=optional_labels(primitive_set.structural_role_labels).astype(np.int32),
        rank_scores=optional_float(primitive_set.rank_scores, (n,)).astype(np.float32),
        gt_match_ids=optional_labels(primitive_set.gt_match_ids).astype(np.int32),
        rank_components=optional_float(primitive_set.rank_components, (n, 0)).astype(np.float32),
        rank_component_names=np.asarray(primitive_set.rank_component_names or (), dtype=object),
        severity_labels=optional_labels(primitive_set.severity_labels).astype(np.int32),
        importance_weights=optional_float(primitive_set.importance_weights, (n,)).astype(np.float32),
    )


def load_primitive_set(path: str | Path) -> GaussianPrimitiveSet:
    data = np.load(path, allow_pickle=True)
    fragment_ids = data["fragment_ids"].astype(np.int32)
    if np.all(fragment_ids < 0):
        fragment_ids = None

    def load_optional(name: str) -> np.ndarray | None:
        if name not in data:
            return None
        values = data[name].astype(np.int32)
        if np.all(values < 0):
            return None
        return values

    def load_optional_float(name: str) -> np.ndarray | None:
        if name not in data:
            return None
        values = data[name].astype(np.float32)
        if values.size == 0 or np.all(np.isnan(values)):
            return None
        return values

    rank_component_names = None
    if "rank_component_names" in data and len(data["rank_component_names"]) > 0:
        rank_component_names = tuple(str(x) for x in data["rank_component_names"].tolist())

    return GaussianPrimitiveSet(
        centers_zyx=data["centers_zyx"].astype(np.float32),
        covariances=data["covariances"].astype(np.float32),
        features=data["features"].astype(np.float32),
        support_counts=data["support_counts"].astype(np.int32),
        feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
        fragment_ids=fragment_ids,
        anatomy_labels=load_optional("anatomy_labels"),
        anatomy_zone_labels=load_optional("anatomy_zone_labels"),
        anatomy_side_labels=load_optional("anatomy_side_labels"),
        structural_role_labels=load_optional("structural_role_labels"),
        rank_scores=load_optional_float("rank_scores"),
        gt_match_ids=load_optional("gt_match_ids"),
        rank_components=load_optional_float("rank_components"),
        rank_component_names=rank_component_names,
        severity_labels=load_optional("severity_labels"),
        importance_weights=load_optional_float("importance_weights"),
    )


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    if len(points) <= n_samples:
        return np.arange(len(points), dtype=np.int64)
    chosen = [int(np.argmin(points[:, 0]))]
    min_dist2 = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, n_samples):
        idx = int(np.argmax(min_dist2))
        chosen.append(idx)
        dist2 = np.sum((points - points[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.asarray(chosen, dtype=np.int64)


def _local_covariance(local_points: np.ndarray, center: np.ndarray) -> np.ndarray:
    if len(local_points) < 3:
        return np.eye(3, dtype=np.float32)
    centered = local_points.astype(np.float32) - center.astype(np.float32)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    cov = cov + np.eye(3, dtype=np.float32) * 1e-3
    return cov.astype(np.float32)


def _primitive_feature_names() -> tuple[str, ...]:
    return (
        "hu_mean",
        "hu_std",
        "cortical_ratio",
        "low_hu_ratio",
        "z_norm",
        "y_norm",
        "x_norm",
        "midline_distance_norm",
        "inferior_norm",
        "side_signed",
    )


def _spacing_feature_names() -> tuple[str, ...]:
    return _primitive_feature_names() + (
        "center_z_mm",
        "center_y_mm",
        "center_x_mm",
    )


def _sacrum_anchor_feature_names() -> tuple[str, ...]:
    return (
        "sacrum_rel_z",
        "sacrum_rel_y",
        "sacrum_rel_x",
        "sacrum_distance_norm",
        "sacrum_side_signed",
    )


def compute_sacrum_anchor_frame(
    labels: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sacrum_range: tuple[int, int] = (1, 50),
) -> SacrumAnchorFrame:
    arr = np.asarray(labels)
    lo, hi = sacrum_range
    mask = (arr >= int(lo)) & (arr <= int(hi))
    if int(mask.sum()) < 1:
        raise ValueError("sacrum anchor requires at least one sacrum voxel")
    points = np.argwhere(mask).astype(np.float32)
    centroid = points.mean(axis=0).astype(np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    extents = np.asarray(arr.shape, dtype=np.float32) * spacing
    norm = max(float(np.max(extents) / 2.0), 1.0)
    return SacrumAnchorFrame(centroid_zyx=centroid, spacing_zyx=spacing, norm_mm=float(norm))


def attach_sacrum_anchor_features(
    primitive_set: GaussianPrimitiveSet,
    frame: SacrumAnchorFrame,
) -> GaussianPrimitiveSet:
    centers = primitive_set.centers_zyx.astype(np.float32)
    rel_mm = (centers - frame.centroid_zyx.astype(np.float32)) * frame.spacing_zyx.astype(np.float32)
    rel_norm = np.clip(rel_mm / max(float(frame.norm_mm), 1e-6), -1.0, 1.0)
    distance = np.linalg.norm(rel_norm, axis=1, keepdims=True)
    side = rel_norm[:, 1:2]
    anchor_features = np.concatenate([rel_norm, distance, side], axis=1).astype(np.float32)
    return replace(
        primitive_set,
        features=np.concatenate([primitive_set.features.astype(np.float32), anchor_features], axis=1),
        feature_names=primitive_set.feature_names + _sacrum_anchor_feature_names(),
    )


def attach_zero_sacrum_anchor_features(primitive_set: GaussianPrimitiveSet) -> GaussianPrimitiveSet:
    zeros = np.zeros((len(primitive_set.centers_zyx), len(_sacrum_anchor_feature_names())), dtype=np.float32)
    return replace(
        primitive_set,
        features=np.concatenate([primitive_set.features.astype(np.float32), zeros], axis=1),
        feature_names=primitive_set.feature_names + _sacrum_anchor_feature_names(),
    )


def _build_primitive_set_from_centers(
    ct: np.ndarray,
    mask_points: np.ndarray,
    centers: np.ndarray,
    radius: float,
    cortical_hu: float,
    low_hu: float,
    spacing_zyx: tuple[float, float, float] | None = None,
    radius_units: str = "voxel",
) -> GaussianPrimitiveSet:
    feature_names = _spacing_feature_names() if radius_units == "mm" else _primitive_feature_names()
    if len(centers) == 0:
        return GaussianPrimitiveSet(
            centers_zyx=np.zeros((0, 3), dtype=np.float32),
            covariances=np.zeros((0, 3, 3), dtype=np.float32),
            features=np.zeros((0, len(feature_names)), dtype=np.float32),
            support_counts=np.zeros((0,), dtype=np.int32),
            feature_names=feature_names,
        )
    spacing = np.asarray(spacing_zyx or (1.0, 1.0, 1.0), dtype=np.float32)
    if radius_units not in {"voxel", "mm"}:
        raise ValueError("radius_units must be 'voxel' or 'mm'")
    tree_points = mask_points * spacing if radius_units == "mm" else mask_points
    tree = cKDTree(tree_points)
    covariances = []
    features = []
    support_counts = []
    shape = np.asarray(ct.shape, dtype=np.float32)
    denom = np.maximum(shape - 1.0, 1.0)
    mid_y = (shape[1] - 1.0) * 0.5
    half_y = max(float(mid_y), 1.0)
    for center in centers.astype(np.float32):
        query_center = center * spacing if radius_units == "mm" else center
        local_idx = tree.query_ball_point(query_center, r=float(radius))
        local_points = mask_points[local_idx] if local_idx else center[None, :]
        local_intensity = ct[tuple(local_points.astype(np.int64).T)]
        center_norm = center / denom
        feature_values = [
            float(np.mean(local_intensity)),
            float(np.std(local_intensity)),
            float(np.mean(local_intensity >= cortical_hu)),
            float(np.mean(local_intensity <= low_hu)),
            float(center_norm[0]),
            float(center_norm[1]),
            float(center_norm[2]),
            float(abs(center[1] - mid_y) / half_y),
            float(center_norm[0]),
            float((center[1] - mid_y) / half_y),
        ]
        if radius_units == "mm":
            center_mm = center * spacing
            feature_values.extend(float(x) for x in center_mm)
        covariances.append(_local_covariance(local_points, center))
        support_counts.append(len(local_points))
        features.append(feature_values)
    return GaussianPrimitiveSet(
        centers_zyx=centers.astype(np.float32),
        covariances=np.asarray(covariances, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        support_counts=np.asarray(support_counts, dtype=np.int32),
        feature_names=feature_names,
    )


def extract_gaussian_primitives(
    ct_hu: np.ndarray,
    foreground_mask: np.ndarray,
    num_primitives: int = 64,
    radius: float = 4.0,
    cortical_hu: float = 300.0,
    low_hu: float = 80.0,
    spacing_zyx: tuple[float, float, float] | None = None,
    radius_units: str = "voxel",
    initial_centers_zyx: np.ndarray | None = None,
) -> GaussianPrimitiveSet:
    """Extract hand-built anisotropic primitives from CT + foreground mask."""
    ct = np.asarray(ct_hu, dtype=np.float32)
    mask = np.asarray(foreground_mask, dtype=bool)
    if ct.shape != mask.shape:
        raise ValueError(f"ct_hu shape {ct.shape} does not match mask shape {mask.shape}")
    points = np.argwhere(mask).astype(np.float32)
    if len(points) == 0:
        return _build_primitive_set_from_centers(
            ct, points, np.zeros((0, 3), dtype=np.float32), radius, cortical_hu, low_hu, spacing_zyx, radius_units
        )

    if initial_centers_zyx is None:
        sample_idx = _farthest_point_sample(points, max(1, int(num_primitives)))
        centers = points[sample_idx]
    else:
        centers = np.asarray(initial_centers_zyx, dtype=np.float32)
    return _build_primitive_set_from_centers(ct, points, centers, radius, cortical_hu, low_hu, spacing_zyx, radius_units)


def extract_gt_balanced_gaussian_primitives(
    ct_hu: np.ndarray,
    gt_instance_labels: np.ndarray,
    num_primitives: int = 128,
    min_primitives_per_fragment: int = 2,
    radius: float = 4.0,
    cortical_hu: float = 300.0,
    low_hu: float = 80.0,
    fragment_importance: dict[int, float] | None = None,
    fragment_severity: dict[int, int] | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    radius_units: str = "voxel",
) -> GaussianPrimitiveSet:
    """Extract primitives with a minimum quota for every GT fragment."""
    ct = np.asarray(ct_hu, dtype=np.float32)
    gt = np.asarray(gt_instance_labels)
    if ct.shape != gt.shape:
        raise ValueError(f"ct_hu shape {ct.shape} does not match gt shape {gt.shape}")
    foreground = gt > 0
    all_points = np.argwhere(foreground).astype(np.float32)
    if len(all_points) == 0:
        return _build_primitive_set_from_centers(
            ct, all_points, np.zeros((0, 3), dtype=np.float32), radius, cortical_hu, low_hu, spacing_zyx, radius_units
        )

    centers = []
    used = set()
    fragment_ids = [int(x) for x in np.unique(gt) if int(x) > 0]
    quota = max(1, int(min_primitives_per_fragment))
    importance = fragment_importance or {}
    severity = fragment_severity or {}
    points_by_fragment = {
        fragment_id: np.argwhere(gt == fragment_id).astype(np.float32)
        for fragment_id in fragment_ids
    }
    # First pass: protect fragment recall by guaranteeing the minimum quota
    # before any severity/importance extras consume the fixed primitive budget.
    for fragment_id in fragment_ids:
        points = points_by_fragment[fragment_id]
        if len(points) == 0:
            continue
        n = min(quota, len(points), max(int(num_primitives) - len(centers), 0))
        if n <= 0:
            break
        selected = points[_farthest_point_sample(points, n)]
        for center in selected:
            key = tuple(int(x) for x in center)
            if key not in used:
                centers.append(center)
                used.add(key)

    extra_budget = max(int(num_primitives) - len(centers), 0)
    extra_allocations = {fragment_id: 0 for fragment_id in fragment_ids}
    if extra_budget > 0 and fragment_ids:
        weights = np.asarray([float(np.clip(importance.get(fid, 1.0), 0.5, 3.0)) for fid in fragment_ids], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-6)
        raw = weights * extra_budget
        base = np.floor(raw).astype(np.int32)
        remainder = int(extra_budget - int(base.sum()))
        if remainder > 0:
            order = np.argsort(raw - base)[::-1]
            for idx in order[:remainder]:
                base[int(idx)] += 1
        extra_allocations = {fragment_id: int(base[idx]) for idx, fragment_id in enumerate(fragment_ids)}

    for fragment_id in fragment_ids:
        points = points_by_fragment[fragment_id]
        if len(points) == 0:
            continue
        n = min(extra_allocations.get(fragment_id, 0), len(points), max(int(num_primitives) - len(centers), 0))
        if n <= 0:
            continue
        selected = points[_farthest_point_sample(points, n)]
        for center in selected:
            key = tuple(int(x) for x in center)
            if key not in used:
                centers.append(center)
                used.add(key)

    remaining = max(int(num_primitives) - len(centers), 0)
    if remaining > 0:
        selected = all_points[_farthest_point_sample(all_points, min(len(all_points), int(num_primitives) + remaining))]
        for center in selected:
            if len(centers) >= int(num_primitives):
                break
            key = tuple(int(x) for x in center)
            if key not in used:
                centers.append(center)
                used.add(key)

    centers_arr = np.asarray(centers[: int(num_primitives)], dtype=np.float32)
    primitive_set = _build_primitive_set_from_centers(ct, all_points, centers_arr, radius, cortical_hu, low_hu, spacing_zyx, radius_units)
    severity_labels = []
    importance_weights = []
    shape = np.asarray(gt.shape, dtype=np.int64)
    for center in primitive_set.centers_zyx:
        idx = np.rint(center).astype(np.int64)
        idx = np.maximum(0, np.minimum(idx, shape - 1))
        fragment_id = int(gt[tuple(idx)])
        severity_labels.append(int(severity.get(fragment_id, 0)))
        importance_weights.append(float(np.clip(importance.get(fragment_id, 1.0), 0.5, 3.0)))
    return replace(
        primitive_set,
        severity_labels=np.asarray(severity_labels, dtype=np.int32),
        importance_weights=np.asarray(importance_weights, dtype=np.float32),
    )


def order_primitives_by_pca_path(primitive_set: GaussianPrimitiveSet) -> np.ndarray:
    centers = primitive_set.centers_zyx.astype(np.float32)
    if len(centers) == 0:
        return np.zeros((0,), dtype=np.int64)
    if len(centers) == 1:
        return np.array([0], dtype=np.int64)
    centered = centers - centers.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    score = centered @ axis
    return np.argsort(score).astype(np.int64)


def assign_primitive_anatomy_priors(
    primitive_set: GaussianPrimitiveSet,
    volume_shape: tuple[int, int, int],
) -> GaussianPrimitiveSet:
    """Assign coarse pelvic anatomy priors from primitive geometry.

    This is a deterministic scaffold, not a final anatomy segmenter. It gives
    the sequence model stable pelvic coordinates before learned priors exist.
    """
    shape = np.asarray(volume_shape, dtype=np.float32)
    mid_y = (shape[1] - 1.0) * 0.5
    mid_tol = max(float(shape[1]) * 0.15, 1.0)
    anatomy = []
    zones = []
    sides = []
    roles = []
    for z, y, _x in primitive_set.centers_zyx:
        lateral = abs(float(y) - mid_y)
        is_midline = lateral <= mid_tol
        is_inferior_lateral = z >= shape[0] * 0.75 and not is_midline

        if is_midline:
            anatomy.append(ANATOMY_LABELS["sacrum"])
            zones.append(ANATOMY_ZONE_LABELS["sacral_body"])
            sides.append(ANATOMY_SIDE_LABELS["midline"])
            roles.append(ANATOMY_STRUCTURAL_ROLES["ring_bridge"])
        elif is_inferior_lateral:
            anatomy.append(ANATOMY_LABELS["femur"])
            zones.append(ANATOMY_ZONE_LABELS["shaft"])
            sides.append(ANATOMY_SIDE_LABELS["left"] if y < mid_y else ANATOMY_SIDE_LABELS["right"])
            roles.append(ANATOMY_STRUCTURAL_ROLES["long_bone_axis"])
        else:
            anatomy.append(ANATOMY_LABELS["pelvic_ring"])
            zones.append(ANATOMY_ZONE_LABELS["ilium"])
            sides.append(ANATOMY_SIDE_LABELS["left"] if y < mid_y else ANATOMY_SIDE_LABELS["right"])
            roles.append(ANATOMY_STRUCTURAL_ROLES["ring_bridge"])

    return replace(
        primitive_set,
        anatomy_labels=np.asarray(anatomy, dtype=np.int32),
        anatomy_zone_labels=np.asarray(zones, dtype=np.int32),
        anatomy_side_labels=np.asarray(sides, dtype=np.int32),
        structural_role_labels=np.asarray(roles, dtype=np.int32),
    )


def order_primitives_by_anatomy_path(primitive_set: GaussianPrimitiveSet) -> np.ndarray:
    """Order primitives by pelvic anatomy before local geometric position."""
    if len(primitive_set.centers_zyx) == 0:
        return np.zeros((0,), dtype=np.int64)
    if primitive_set.anatomy_labels is None:
        return order_primitives_by_pca_path(primitive_set)
    rank = {
        ANATOMY_LABELS["sacrum"]: 0,
        ANATOMY_LABELS["pelvic_ring"]: 1,
        ANATOMY_LABELS["acetabulum_hip"]: 2,
        ANATOMY_LABELS["femur"]: 3,
        ANATOMY_LABELS["unknown"]: 9,
    }
    side_rank = {
        ANATOMY_SIDE_LABELS["midline"]: 0,
        ANATOMY_SIDE_LABELS["left"]: 1,
        ANATOMY_SIDE_LABELS["right"]: 2,
        ANATOMY_SIDE_LABELS["unknown"]: 3,
    }
    side_labels = (
        primitive_set.anatomy_side_labels
        if primitive_set.anatomy_side_labels is not None
        else np.full(len(primitive_set.centers_zyx), ANATOMY_SIDE_LABELS["unknown"], dtype=np.int32)
    )
    keys = []
    for idx, center in enumerate(primitive_set.centers_zyx):
        keys.append(
            (
                rank.get(int(primitive_set.anatomy_labels[idx]), 9),
                side_rank.get(int(side_labels[idx]), 3),
                float(center[0]),
                float(center[1]),
                float(center[2]),
                idx,
            )
        )
    return np.asarray([idx for idx, _ in sorted(enumerate(keys), key=lambda item: item[1])], dtype=np.int64)


def assign_primitive_fragment_ids(
    primitive_set: GaussianPrimitiveSet,
    gt_instance_labels: np.ndarray,
) -> GaussianPrimitiveSet:
    labels = np.asarray(gt_instance_labels)
    shape = np.asarray(labels.shape, dtype=np.int64)
    ids = []
    for center in primitive_set.centers_zyx:
        idx = np.rint(center).astype(np.int64)
        idx = np.maximum(0, np.minimum(idx, shape - 1))
        ids.append(int(labels[tuple(idx)]))
    return replace(primitive_set, fragment_ids=np.asarray(ids, dtype=np.int32))


def build_primitive_affinity_labels(
    primitive_set: GaussianPrimitiveSet,
    k_nearest: int = 4,
) -> list[PrimitiveAffinityEdge]:
    if primitive_set.fragment_ids is None:
        raise ValueError("fragment_ids are required to build primitive affinity labels")
    centers = primitive_set.centers_zyx
    if len(centers) <= 1:
        return []
    tree = cKDTree(centers)
    k = min(len(centers), max(2, int(k_nearest) + 1))
    distances, indices = tree.query(centers, k=k)
    edges = set()
    out: list[PrimitiveAffinityEdge] = []
    for i in range(len(centers)):
        for dist, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(indices[i]), strict=False):
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges:
                continue
            edges.add((a, b))
            ida, idb = int(primitive_set.fragment_ids[a]), int(primitive_set.fragment_ids[b])
            if ida <= 0 or idb <= 0:
                continue
            out.append(PrimitiveAffinityEdge(a, b, ida == idb, float(dist)))
    return out


def decode_primitive_affinities(
    edges: np.ndarray,
    attractive: np.ndarray,
    repulsive: np.ndarray,
    n_primitives: int,
    attractive_threshold: float = 0.5,
    repulsive_threshold: float = 0.5,
) -> np.ndarray:
    """Constrained union-find decode for primitive graph affinities."""
    parent = list(range(int(n_primitives)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    mutex: set[tuple[int, int]] = set()
    events = []
    for idx, (i, j) in enumerate(np.asarray(edges, dtype=np.int32)):
        events.append((float(attractive[idx]), "a", int(i), int(j)))
        events.append((float(repulsive[idx]), "r", int(i), int(j)))
    for score, kind, i, j in sorted(events, key=lambda item: item[0], reverse=True):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        key = (ri, rj) if ri < rj else (rj, ri)
        if kind == "r" and score >= repulsive_threshold:
            mutex.add(key)
        elif kind == "a" and score >= attractive_threshold and key not in mutex:
            union(ri, rj)
            mutex = {
                ((find(a), find(b)) if find(a) < find(b) else (find(b), find(a)))
                for a, b in mutex
                if find(a) != find(b)
            }

    root_to_label: dict[int, int] = {}
    labels = np.zeros((n_primitives,), dtype=np.int32)
    next_label = 1
    for idx in range(n_primitives):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[idx] = root_to_label[root]
    return labels


def splat_primitives_to_voxels(
    primitive_set: GaussianPrimitiveSet,
    primitive_labels: np.ndarray,
    foreground_mask: np.ndarray,
    target_range: tuple[int, int] = (1, 255),
) -> np.ndarray:
    """Assign each foreground voxel to the nearest primitive label."""
    mask = np.asarray(foreground_mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.uint16)
    if len(primitive_set.centers_zyx) == 0 or not mask.any():
        return out
    tree = cKDTree(primitive_set.centers_zyx)
    voxels = np.argwhere(mask).astype(np.float32)
    _, nearest = tree.query(voxels, k=1)
    lo, hi = target_range
    label_to_id: dict[int, int] = {}
    next_id = lo
    for voxel, prim_idx in zip(voxels.astype(np.int64), nearest, strict=False):
        label = int(primitive_labels[int(prim_idx)])
        if label not in label_to_id:
            if next_id > hi:
                continue
            label_to_id[label] = next_id
            next_id += 1
        out[tuple(voxel)] = label_to_id[label]
    return out
