"""Small Mutex-Watershed-style decoder prototype.

This is a research decoder, not a production-optimized implementation. It uses
local attractive edges to union foreground voxels while high-confidence
repulsive edges register mutex constraints between components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scripts.affinity_labels import AffinityOffset


@dataclass(frozen=True)
class _Edge:
    score: float
    kind: str
    a: int
    b: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def _source_destination_indices(
    shape: tuple[int, int, int],
    offset: AffinityOffset,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    dz, dy, dx = offset.as_tuple()
    z, y, x = shape
    src_slices = (
        slice(0, z - dz if dz else z),
        slice(0, y - dy if dy else y),
        slice(0, x - dx if dx else x),
    )
    dst_slices = (
        slice(dz, z),
        slice(dy, y),
        slice(dx, x),
    )
    flat = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    return flat[src_slices], flat[dst_slices], src_slices


def mutex_watershed_decode(
    foreground: np.ndarray,
    attractive: np.ndarray,
    repulsive: np.ndarray,
    offsets: Iterable[AffinityOffset],
    attractive_threshold: float = 0.5,
    repulsive_threshold: float = 0.5,
    target_range: tuple[int, int] = (1, 255),
) -> np.ndarray:
    """Decode foreground instances from local attractive/repulsive affinities."""
    fg = np.asarray(foreground, dtype=bool)
    if fg.ndim != 3:
        raise ValueError(f"foreground must be 3D, got shape {fg.shape}")
    chosen_offsets = tuple(offsets)
    expected = (len(chosen_offsets), *fg.shape)
    if attractive.shape != expected or repulsive.shape != expected:
        raise ValueError(
            f"affinity arrays must have shape {expected}, got {attractive.shape} and {repulsive.shape}"
        )

    n_voxels = int(np.prod(fg.shape))
    uf = _UnionFind(n_voxels)
    edges: list[_Edge] = []

    for offset_idx, offset in enumerate(chosen_offsets):
        src_flat, dst_flat, src_slices = _source_destination_indices(fg.shape, offset)
        valid = fg[src_slices] & fg.reshape(fg.shape)[
            slice(offset.dz, fg.shape[0]),
            slice(offset.dy, fg.shape[1]),
            slice(offset.dx, fg.shape[2]),
        ]
        if not valid.any():
            continue
        att = attractive[(offset_idx, *src_slices)]
        rep = repulsive[(offset_idx, *src_slices)]
        for src, dst, score in zip(src_flat[valid], dst_flat[valid], att[valid], strict=False):
            if float(score) >= attractive_threshold:
                edges.append(_Edge(float(score), "attractive", int(src), int(dst)))
        for src, dst, score in zip(src_flat[valid], dst_flat[valid], rep[valid], strict=False):
            if float(score) >= repulsive_threshold:
                edges.append(_Edge(float(score), "repulsive", int(src), int(dst)))

    mutex: set[tuple[int, int]] = set()
    for edge in sorted(edges, key=lambda item: item.score, reverse=True):
        ra, rb = uf.find(edge.a), uf.find(edge.b)
        if ra == rb:
            continue
        key = (ra, rb) if ra < rb else (rb, ra)
        if edge.kind == "repulsive":
            mutex.add(key)
            continue
        if key in mutex:
            continue
        new_root = uf.union(ra, rb)
        old_roots = {ra, rb}
        rewritten: set[tuple[int, int]] = set()
        for ma, mb in mutex:
            na = new_root if ma in old_roots else uf.find(ma)
            nb = new_root if mb in old_roots else uf.find(mb)
            if na == nb:
                continue
            rewritten.add((na, nb) if na < nb else (nb, na))
        mutex = rewritten

    labels = np.zeros(fg.shape, dtype=np.uint16)
    lo, hi = target_range
    next_id = lo
    root_to_id: dict[int, int] = {}
    flat_fg = fg.ravel()
    flat_labels = labels.ravel()
    for flat_idx in np.flatnonzero(flat_fg):
        root = uf.find(int(flat_idx))
        if root not in root_to_id:
            if next_id > hi:
                break
            root_to_id[root] = next_id
            next_id += 1
        flat_labels[flat_idx] = root_to_id[root]
    return labels
