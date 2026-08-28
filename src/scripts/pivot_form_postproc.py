"""PIVOT-Form v0 — fragment formation on border-core predictions.

Replaces naive border-adjacency CC counting with:
  1. volume filter + border-adjacency (same as legacy)
  2. intact-case cardinality guard (low border/core ratio → single fragment)
  3. greedy graph merge on core CC pairs (border bridge + size + distance)

See docs/PIVOT_FORM_SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass, grey_dilation
from scipy.ndimage import label as cc_label

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class FormConfig:
    min_volume_mm3: float = 500.0
    tau_intact: float = 0.08
    theta_merge: float = 0.35
    alpha_border: float = 1.0
    beta_size: float = 0.5
    gamma_dist: float = 0.3
    d_max_mm: float = 80.0
    border_dilation_iters: int = 2


def _voxel_volume_mm3(spacing_xyz: tuple[float, float, float]) -> float:
    return float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])


def border_core_ratio(
    core_mask: np.ndarray,
    border_mask: np.ndarray,
    dilation_iters: int = 2,
) -> float:
    n_core = int(core_mask.sum())
    if n_core == 0:
        return float("nan")
    band = (
        binary_dilation(core_mask, structure=STRUCT26, iterations=dilation_iters)
        & border_mask
    )
    return int(band.sum()) / n_core


def _core_components(
    core_mask: np.ndarray,
    voxel_volume_mm3: float,
    min_volume_mm3: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (label map, [(cid, n_voxels), ...]) for CCs above min volume."""
    labels, n = cc_label(core_mask, structure=STRUCT26)
    if n == 0:
        return labels, []
    kept: list[tuple[int, int]] = []
    for cid in range(1, n + 1):
        nv = int((labels == cid).sum())
        if nv * voxel_volume_mm3 >= min_volume_mm3:
            kept.append((cid, nv))
    return labels, kept


def _border_adjacency_filter(
    labels: np.ndarray,
    kept: list[tuple[int, int]],
    border_mask: np.ndarray,
    dilation_iters: int,
) -> list[tuple[int, int]]:
    if not kept:
        return []
    largest_cid = max(kept, key=lambda x: x[1])[0]
    if not border_mask.any():
        return [x for x in kept if x[0] == largest_cid]

    out: list[tuple[int, int]] = []
    for cid, nv in kept:
        if cid == largest_cid:
            out.append((cid, nv))
            continue
        cc = labels == cid
        adj = (
            binary_dilation(cc, structure=STRUCT26, iterations=dilation_iters)
            & border_mask
        )
        if adj.any():
            out.append((cid, nv))
    return out


def _centroid_mm(
    labels: np.ndarray,
    cid: int,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    # scipy center_of_mass uses array index order (z, y, x)
    com = center_of_mass(labels == cid)
    return np.array([com[2] * spacing_zyx[2], com[1] * spacing_zyx[1], com[0] * spacing_zyx[0]])


def _merge_score(
    labels: np.ndarray,
    cid_i: int,
    cid_j: int,
    vol_i: int,
    vol_j: int,
    border_mask: np.ndarray,
    dist_mm: float,
    cfg: FormConfig,
) -> float:
    if dist_mm > cfg.d_max_mm:
        return -1.0
    mi = labels == cid_i
    mj = labels == cid_j
    touch = (
        binary_dilation(mi | mj, structure=STRUCT26, iterations=cfg.border_dilation_iters)
        & border_mask
    )
    bridge = int(touch.sum()) / max(min(vol_i, vol_j), 1)
    size_term = float(np.exp(-abs(np.log(max(vol_i, 1)) - np.log(max(vol_j, 1)))))
    dist_term = dist_mm / max(cfg.d_max_mm, 1e-6)
    return (
        cfg.alpha_border * bridge
        + cfg.beta_size * size_term
        - cfg.gamma_dist * dist_term
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _greedy_merge_components(
    labels: np.ndarray,
    kept: list[tuple[int, int]],
    border_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cfg: FormConfig,
) -> list[tuple[int, int]]:
    """Merge CC list via union-find; return merged [(representative_cid, total_voxels), ...]."""
    if len(kept) <= 1:
        return kept

    cids = [c for c, _ in kept]
    vols = {c: v for c, v in kept}
    cents = {c: _centroid_mm(labels, c, spacing_zyx) for c in cids}
    idx = {c: i for i, c in enumerate(cids)}

    edges: list[tuple[float, int, int]] = []
    for i, ci in enumerate(cids):
        for cj in cids[i + 1 :]:
            d = float(np.linalg.norm(cents[ci] - cents[cj]))
            s = _merge_score(labels, ci, cj, vols[ci], vols[cj], border_mask, d, cfg)
            if s >= 0:
                edges.append((s, idx[ci], idx[cj]))
    edges.sort(reverse=True)

    uf = _UnionFind(len(cids))
    for s, a, b in edges:
        if s < cfg.theta_merge:
            break
        uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for i, c in enumerate(cids):
        r = uf.find(i)
        groups.setdefault(r, []).append(c)

    merged: list[tuple[int, int]] = []
    for members in groups.values():
        total = sum(vols[c] for c in members)
        rep = max(members, key=lambda c: vols[c])
        merged.append((rep, total))
    return merged


def select_fragments_form(
    core_mask: np.ndarray,
    border_mask: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cfg: FormConfig | None = None,
) -> list[tuple[int, int]]:
    """Return [(cc_id, n_voxels), ...] after PIVOT-Form v0 (not yet mapped to PENGWIN IDs)."""
    cfg = cfg or FormConfig()
    labels, kept = _core_components(core_mask, voxel_volume_mm3, cfg.min_volume_mm3)
    if not kept:
        return []

    kept = _border_adjacency_filter(labels, kept, border_mask, cfg.border_dilation_iters)
    if not kept:
        return []

    ratio = border_core_ratio(core_mask, border_mask, cfg.border_dilation_iters)
    if len(kept) >= 2 and ratio == ratio and ratio < cfg.tau_intact:
        return [max(kept, key=lambda x: x[1])]

    if len(kept) >= 2:
        kept = _greedy_merge_components(labels, kept, border_mask, spacing_zyx, cfg)

    kept.sort(key=lambda x: x[1], reverse=True)
    return kept


def expand_border_into_instances(
    inst_core: np.ndarray,
    border_mask: np.ndarray,
    dilation_iters: int = 3,
) -> np.ndarray:
    if not border_mask.any():
        return inst_core
    expanded = inst_core.copy()
    for _ in range(dilation_iters):
        dilated = grey_dilation(expanded, footprint=STRUCT26)
        fill = (expanded == 0) & border_mask & (dilated > 0)
        expanded[fill] = dilated[fill]
    return expanded


def border_core_to_instances(
    semantic: np.ndarray,
    core_label: int,
    border_label: int,
    target_range: tuple[int, int],
    voxel_volume_mm3: float,
    min_volume_mm3: float = 500.0,
    form_cfg: FormConfig | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """7-class semantic → PENGWIN instance IDs for one anatomy.

    form_cfg is None → legacy border-adjacency only (no guard / graph merge).
    """
    out = np.zeros_like(semantic, dtype=np.uint16)
    core_mask = semantic == core_label
    border_mask = semantic == border_label
    if not core_mask.any():
        return out

    labels, n = cc_label(core_mask, structure=STRUCT26)
    if n == 0:
        return out

    if form_cfg is None:
        fragments = _legacy_fragments(
            labels, border_mask, voxel_volume_mm3, min_volume_mm3
        )
    else:
        cfg = FormConfig(
            min_volume_mm3=min_volume_mm3,
            tau_intact=form_cfg.tau_intact,
            theta_merge=form_cfg.theta_merge,
            alpha_border=form_cfg.alpha_border,
            beta_size=form_cfg.beta_size,
            gamma_dist=form_cfg.gamma_dist,
            d_max_mm=form_cfg.d_max_mm,
            border_dilation_iters=form_cfg.border_dilation_iters,
        )
        fragments = select_fragments_form(
            core_mask, border_mask, voxel_volume_mm3, spacing_zyx, cfg
        )

    lo, hi = target_range
    max_slots = hi - lo + 1
    fragments = fragments[:max_slots]

    for slot, (cid, _) in enumerate(fragments):
        out[labels == cid] = lo + slot

    return expand_border_into_instances(out, border_mask)


def _legacy_fragments(
    labels: np.ndarray,
    border_mask: np.ndarray,
    voxel_volume_mm3: float,
    min_volume_mm3: float,
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    n = int(labels.max())
    for cid in range(1, n + 1):
        nv = int((labels == cid).sum())
        if nv * voxel_volume_mm3 >= min_volume_mm3:
            kept.append((cid, nv))
    return _border_adjacency_filter(labels, kept, border_mask, 2)


def assemble_pelvic_instances(
    semantic_7class: np.ndarray,
    voxel_volume_mm3: float,
    anatomy_map: dict[str, tuple[int, int]],
    target_ranges: dict[str, tuple[int, int]],
    form_cfg: FormConfig | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    out = np.zeros_like(semantic_7class, dtype=np.uint16)
    for name, (core_l, border_l) in anatomy_map.items():
        ids = border_core_to_instances(
            semantic_7class,
            core_l,
            border_l,
            target_ranges[name],
            voxel_volume_mm3,
            min_volume_mm3=form_cfg.min_volume_mm3 if form_cfg else 500.0,
            form_cfg=form_cfg,
            spacing_zyx=spacing_zyx,
        )
        out += ids
    return out
