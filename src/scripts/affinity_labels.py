"""Local affinity targets for fracture instance segmentation.

For each offset, an edge is valid only when both endpoint voxels are foreground.
Valid same-instance edges are attractive; valid different-instance edges are
repulsive. Arrays keep the same spatial shape as the source label volume, with
invalid border/background edges set to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AffinityOffset:
    dz: int
    dy: int
    dx: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.dz), int(self.dy), int(self.dx))


DEFAULT_AFFINITY_OFFSETS: tuple[AffinityOffset, ...] = (
    AffinityOffset(1, 0, 0),
    AffinityOffset(0, 1, 0),
    AffinityOffset(0, 0, 1),
    AffinityOffset(2, 0, 0),
    AffinityOffset(0, 2, 0),
    AffinityOffset(0, 0, 2),
    AffinityOffset(4, 0, 0),
    AffinityOffset(0, 4, 0),
    AffinityOffset(0, 0, 4),
)


@dataclass(frozen=True)
class AffinityTargets:
    offsets: tuple[AffinityOffset, ...]
    attractive: np.ndarray
    repulsive: np.ndarray
    valid: np.ndarray


def _slices_for_offset(
    shape: tuple[int, int, int],
    offset: AffinityOffset,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    dz, dy, dx = offset.as_tuple()
    if dz < 0 or dy < 0 or dx < 0:
        raise ValueError(f"offsets must be non-negative, got {offset}")
    if dz == 0 and dy == 0 and dx == 0:
        raise ValueError("zero offset is not a valid affinity edge")
    z, y, x = shape
    src = (
        slice(0, z - dz if dz else z),
        slice(0, y - dy if dy else y),
        slice(0, x - dx if dx else x),
    )
    dst = (
        slice(dz, z),
        slice(dy, y),
        slice(dx, x),
    )
    return src, dst


def build_affinity_targets(
    instance_labels: np.ndarray,
    offsets: Iterable[AffinityOffset] = DEFAULT_AFFINITY_OFFSETS,
    foreground_ids: set[int] | None = None,
) -> AffinityTargets:
    """Build attractive/repulsive local-edge targets from instance labels.

    Parameters
    ----------
    instance_labels:
        3D uint label map. Background must be 0.
    offsets:
        Non-negative z/y/x offsets. Each offset compares `voxel` to
        `voxel + offset`.
    foreground_ids:
        Optional set of IDs to include. Other IDs are treated as background.
    """
    labels = np.asarray(instance_labels)
    if labels.ndim != 3:
        raise ValueError(f"instance_labels must be 3D, got shape {labels.shape}")
    chosen_offsets = tuple(offsets)
    shape = labels.shape
    n_offsets = len(chosen_offsets)
    attractive = np.zeros((n_offsets, *shape), dtype=np.float32)
    repulsive = np.zeros((n_offsets, *shape), dtype=np.float32)
    valid = np.zeros((n_offsets, *shape), dtype=bool)

    if foreground_ids is None:
        foreground = labels > 0
    else:
        foreground = np.isin(labels, list(foreground_ids))

    for idx, offset in enumerate(chosen_offsets):
        src, dst = _slices_for_offset(shape, offset)
        src_fg = foreground[src]
        dst_fg = foreground[dst]
        edge_valid = src_fg & dst_fg
        same = edge_valid & (labels[src] == labels[dst])
        different = edge_valid & (labels[src] != labels[dst])
        valid[(idx, *src)] = edge_valid
        attractive[(idx, *src)] = same.astype(np.float32)
        repulsive[(idx, *src)] = different.astype(np.float32)

    return AffinityTargets(
        offsets=chosen_offsets,
        attractive=attractive,
        repulsive=repulsive,
        valid=valid,
    )
