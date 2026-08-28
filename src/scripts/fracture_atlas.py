"""Severity-aware fracture atlas utilities for Gaussian Mamba."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


FRACTURE_SEVERITY_LABELS: dict[str, int] = {
    "unknown": 0,
    "micro_fracture": 1,
    "minor_fracture": 2,
    "major_fracture": 3,
    "complex_comminuted": 4,
}


@dataclass(frozen=True)
class FractureAtlasConfig:
    micro_max_voxels: int = 16
    minor_max_voxels: int = 256
    complex_fragment_count: int = 16
    min_importance: float = 0.5
    max_importance: float = 3.0
    micro_weight: float = 2.0
    minor_weight: float = 1.4
    major_weight: float = 1.0
    complex_weight: float = 1.8
    difficulty_weight: float = 1.5
    rarity_weight: float = 1.25


@dataclass(frozen=True)
class FragmentAtlasStats:
    case_id: str
    fragment_id: int
    volume_voxels: int
    centroid_zyx: tuple[float, float, float]
    bbox_zyx: tuple[int, int, int]
    severity: str
    severity_label: int
    rarity_weight: float
    difficulty_weight: float
    importance_weight: float


@dataclass(frozen=True)
class FractureAtlas:
    fragments: tuple[FragmentAtlasStats, ...]
    zone_summary: dict[str, Any]
    num_cases: int
    config: FractureAtlasConfig


def classify_fragment_severity(
    volume_voxels: int,
    case_fragment_count: int,
    config: FractureAtlasConfig = FractureAtlasConfig(),
) -> str:
    if case_fragment_count >= config.complex_fragment_count:
        return "complex_comminuted"
    if volume_voxels <= config.micro_max_voxels:
        return "micro_fracture"
    if volume_voxels <= config.minor_max_voxels:
        return "minor_fracture"
    return "major_fracture"


def _severity_weight(severity: str, config: FractureAtlasConfig) -> float:
    return {
        "micro_fracture": config.micro_weight,
        "minor_fracture": config.minor_weight,
        "major_fracture": config.major_weight,
        "complex_comminuted": config.complex_weight,
    }.get(severity, 1.0)


def _difficulty_from_ranking(case_id: str, fragment_id: int, ranking_failures: dict[str, set[int]] | None, config: FractureAtlasConfig) -> float:
    if ranking_failures is None:
        return 1.0
    if int(fragment_id) in ranking_failures.get(str(case_id), set()):
        return config.difficulty_weight
    return 1.0


def build_fracture_atlas_from_labels(
    labels_by_case: dict[str, np.ndarray],
    config: FractureAtlasConfig = FractureAtlasConfig(),
    ranking_failures: dict[str, set[int]] | None = None,
) -> FractureAtlas:
    raw_fragments = []
    volumes = []
    for case_id, labels in labels_by_case.items():
        gt = np.asarray(labels)
        fragment_ids = [int(x) for x in np.unique(gt) if int(x) > 0]
        case_fragment_count = len(fragment_ids)
        for fragment_id in fragment_ids:
            points = np.argwhere(gt == fragment_id)
            volume = int(len(points))
            if volume == 0:
                continue
            lo = points.min(axis=0)
            hi = points.max(axis=0)
            bbox = tuple(int(x) for x in (hi - lo + 1))
            centroid = tuple(float(x) for x in points.mean(axis=0))
            severity = classify_fragment_severity(volume, case_fragment_count, config)
            raw_fragments.append((str(case_id), fragment_id, volume, centroid, bbox, severity))
            volumes.append(volume)

    max_volume = max(volumes, default=1)
    fragments = []
    for case_id, fragment_id, volume, centroid, bbox, severity in raw_fragments:
        rarity = float(np.clip(1.0 + config.rarity_weight * (1.0 - volume / max_volume), config.min_importance, config.max_importance))
        difficulty = float(np.clip(_difficulty_from_ranking(case_id, fragment_id, ranking_failures, config), config.min_importance, config.max_importance))
        importance = _severity_weight(severity, config) * rarity * difficulty
        importance = float(np.clip(importance, config.min_importance, config.max_importance))
        fragments.append(
            FragmentAtlasStats(
                case_id=case_id,
                fragment_id=fragment_id,
                volume_voxels=volume,
                centroid_zyx=centroid,
                bbox_zyx=bbox,
                severity=severity,
                severity_label=FRACTURE_SEVERITY_LABELS[severity],
                rarity_weight=float(rarity),
                difficulty_weight=float(difficulty),
                importance_weight=importance,
            )
        )

    severity_counts: dict[str, int] = {}
    for fragment in fragments:
        severity_counts[fragment.severity] = severity_counts.get(fragment.severity, 0) + 1
    return FractureAtlas(
        fragments=tuple(fragments),
        zone_summary={"severity_counts": severity_counts},
        num_cases=len(labels_by_case),
        config=config,
    )


def save_fracture_atlas(path: str | Path, atlas: FractureAtlas) -> None:
    payload = {
        "num_cases": atlas.num_cases,
        "config": asdict(atlas.config),
        "zone_summary": atlas.zone_summary,
        "fragments": [asdict(fragment) for fragment in atlas.fragments],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-npy", nargs="+", required=True, help="Pairs like case_id=/path/gt.npy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels_by_case = {}
    for item in args.gt_npy:
        case_id, path = item.split("=", 1)
        labels_by_case[case_id] = np.load(path)
    atlas = build_fracture_atlas_from_labels(labels_by_case)
    save_fracture_atlas(args.output, atlas)
    print(f"wrote {args.output} with {len(atlas.fragments)} fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
