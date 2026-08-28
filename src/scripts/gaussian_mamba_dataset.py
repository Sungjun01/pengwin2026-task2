"""Build model-ready Gaussian-Mamba samples from ranked primitive artifacts."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gaussian_primitives import (
    ANATOMY_LABELS,
    ANATOMY_SIDE_LABELS,
    ANATOMY_STRUCTURAL_ROLES,
    ANATOMY_ZONE_LABELS,
    GaussianPrimitiveSet,
    load_primitive_set,
    order_primitives_by_anatomy_path,
)
from scripts.gaussian_mamba_instance_decoder import build_seed_membership_targets


@dataclass(frozen=True)
class GaussianMambaTrainingSample:
    features: np.ndarray
    centers_zyx: np.ndarray
    edge_index: np.ndarray
    edge_same_fragment: np.ndarray
    edge_weight: np.ndarray
    targets: dict[str, np.ndarray]
    feature_names: tuple[str, ...]
    order: np.ndarray


def _ordered_or_default(values: np.ndarray | None, order: np.ndarray, default: int) -> np.ndarray:
    if values is None:
        return np.full(len(order), default, dtype=np.int32)
    return values[order].astype(np.int32)


def _build_knn_edges(centers: np.ndarray, k_nearest: int) -> np.ndarray:
    if len(centers) <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    tree = cKDTree(centers)
    k = min(len(centers), max(2, int(k_nearest) + 1))
    _, indices = tree.query(centers, k=k)
    edges: set[tuple[int, int]] = set()
    for i in range(len(centers)):
        for j in np.atleast_1d(indices[i]):
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int32)


def build_mamba_training_sample(
    primitive_set: GaussianPrimitiveSet,
    k_nearest: int = 4,
) -> GaussianMambaTrainingSample:
    if primitive_set.rank_scores is None:
        raise ValueError("rank_scores are required to build Gaussian-Mamba training samples")
    if primitive_set.gt_match_ids is None:
        raise ValueError("gt_match_ids are required to build primitive affinity targets")

    order = order_primitives_by_anatomy_path(primitive_set)
    centers = primitive_set.centers_zyx[order].astype(np.float32)
    rank_scores = primitive_set.rank_scores[order].astype(np.float32)
    features = primitive_set.features[order].astype(np.float32)
    feature_names = tuple(primitive_set.feature_names)

    gt_match_ids = primitive_set.gt_match_ids[order].astype(np.int32)
    importance = (
        primitive_set.importance_weights[order].astype(np.float32)
        if primitive_set.importance_weights is not None
        else np.ones(len(order), dtype=np.float32)
    )
    edge_index = _build_knn_edges(centers, k_nearest=k_nearest)
    same = []
    edge_weight = []
    for i, j in edge_index:
        ida, idb = int(gt_match_ids[i]), int(gt_match_ids[j])
        is_same = 1 if ida > 0 and ida == idb else 0
        same.append(is_same)
        if is_same:
            weight = (importance[i] + importance[j]) * 0.5
        else:
            # Negative edges touching important fragments are the merge-failure
            # cases we most want the affinity head to repel.
            weight = max(float(importance[i]), float(importance[j])) * 1.35
        edge_weight.append(float(np.clip(weight, 0.5, 3.0)))

    targets = {
        "anatomy": _ordered_or_default(primitive_set.anatomy_labels, order, ANATOMY_LABELS["unknown"]),
        "zone": _ordered_or_default(primitive_set.anatomy_zone_labels, order, ANATOMY_ZONE_LABELS["unknown"]),
        "side": _ordered_or_default(primitive_set.anatomy_side_labels, order, ANATOMY_SIDE_LABELS["unknown"]),
        "structural_role": _ordered_or_default(
            primitive_set.structural_role_labels,
            order,
            ANATOMY_STRUCTURAL_ROLES["unknown"],
        ),
        "rank_score": rank_scores,
        "gt_match_id": gt_match_ids,
        "severity": _ordered_or_default(primitive_set.severity_labels, order, 0),
        "importance_weight": importance,
    }
    membership_targets = build_seed_membership_targets(gt_match_ids, rank_scores, max_queries=16)
    targets["instance_fragment_id"] = membership_targets.fragment_ids
    targets["instance_membership"] = membership_targets.membership
    targets["instance_valid"] = membership_targets.valid_query.astype(np.float32)

    return GaussianMambaTrainingSample(
        features=features,
        centers_zyx=centers,
        edge_index=edge_index,
        edge_same_fragment=np.asarray(same, dtype=np.int32),
        edge_weight=np.asarray(edge_weight, dtype=np.float32),
        targets=targets,
        feature_names=feature_names,
        order=order,
    )


def save_mamba_training_sample(path: str | Path, sample: GaussianMambaTrainingSample) -> None:
    np.savez_compressed(
        path,
        features=sample.features.astype(np.float32),
        centers_zyx=sample.centers_zyx.astype(np.float32),
        edge_index=sample.edge_index.astype(np.int32),
        edge_same_fragment=sample.edge_same_fragment.astype(np.int32),
        edge_weight=sample.edge_weight.astype(np.float32),
        feature_names=np.asarray(sample.feature_names, dtype=object),
        order=sample.order.astype(np.int64),
        target_anatomy=sample.targets["anatomy"].astype(np.int32),
        target_zone=sample.targets["zone"].astype(np.int32),
        target_side=sample.targets["side"].astype(np.int32),
        target_structural_role=sample.targets["structural_role"].astype(np.int32),
        target_rank_score=sample.targets["rank_score"].astype(np.float32),
        target_gt_match_id=sample.targets["gt_match_id"].astype(np.int32),
        target_severity=sample.targets["severity"].astype(np.int32),
        target_importance_weight=sample.targets["importance_weight"].astype(np.float32),
        target_instance_fragment_id=sample.targets["instance_fragment_id"].astype(np.int32),
        target_instance_membership=sample.targets["instance_membership"].astype(np.float32),
        target_instance_valid=sample.targets["instance_valid"].astype(np.float32),
    )


def convert_ranked_primitives_to_sample(
    primitive_path: str | Path,
    output_path: str | Path,
    k_nearest: int = 4,
) -> GaussianMambaTrainingSample:
    primitive_set = load_primitive_set(primitive_path)
    sample = build_mamba_training_sample(primitive_set, k_nearest=k_nearest)
    save_mamba_training_sample(output_path, sample)
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", required=True, help="Ranked primitive .npz")
    parser.add_argument("--output", required=True, help="Output model-ready sample .npz")
    parser.add_argument("--k-nearest", type=int, default=4)
    args = parser.parse_args()
    sample = convert_ranked_primitives_to_sample(args.primitives, args.output, k_nearest=args.k_nearest)
    print(
        f"wrote {args.output} with {sample.features.shape[0]} primitives, "
        f"{sample.edge_index.shape[0]} edges, {sample.features.shape[1]} features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
