"""Tier B postproc: keep-all CC + learned merge (§4A.8.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, grey_dilation
from scipy.ndimage import label as cc_label

from scripts.form_learn_edge import EDGE_FEATURE_COLS, cc_kept, contact_edge_pairs, edge_geometry
from scripts.pivot_form_postproc import (
    STRUCT26,
    FormConfig,
    _UnionFind,
    expand_border_into_instances,
)

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = _ROOT / "progress" / "form_learn_merge.json"


def fragments_hybrid_legacy_keep(
    core_mask: np.ndarray,
    border_mask: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float],
    min_volume_mm3: float,
) -> list[tuple[int, int]]:
    """Use keep-all only when legacy border filter drops more CCs than raw volume filter."""
    from scripts.pivot_form_postproc import _border_adjacency_filter, _core_components

    labels, kept = _core_components(core_mask, voxel_volume_mm3, min_volume_mm3)
    if not kept:
        return []
    legacy_kept = _border_adjacency_filter(labels, kept, border_mask, 2)
    if len(legacy_kept) < len(kept):
        return [(c, nv) for c, nv in kept]
    return legacy_kept


def fragments_keep_all(
    core_mask: np.ndarray,
    voxel_volume_mm3: float,
    min_volume_mm3: float,
) -> list[tuple[int, int]]:
    labels, kept, sizes = cc_kept(core_mask, voxel_volume_mm3, min_volume_mm3)
    return [(c, int(sizes[c - 1])) for c in kept]


def fragments_learned_merge(
    core_mask: np.ndarray,
    border_mask: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float],
    min_volume_mm3: float,
    merge_weights: dict,
    theta_merge: float = 0.35,
) -> list[tuple[int, int]]:
    """Keep all volume CCs, then union-find merge edges above threshold."""
    labels, kept, sizes = cc_kept(core_mask, voxel_volume_mm3, min_volume_mm3)
    if len(kept) <= 1:
        return [(c, int(sizes[c - 1])) for c in kept]

    cids = kept
    idx = {c: i for i, c in enumerate(cids)}
    vols = {c: int(sizes[c - 1]) for c in cids}

    w = merge_weights.get("weights", [])
    bias = float(merge_weights.get("bias", 0.0))
    scale = merge_weights.get("feature_means", {})
    std = merge_weights.get("feature_stds", {})

    edges_scored: list[tuple[float, int, int]] = []
    for ci, cj in contact_edge_pairs(labels, kept, border_mask):
        geom = edge_geometry(labels, ci, cj, border_mask, voxel_volume_mm3, spacing_zyx)
        feats = []
        for col in EDGE_FEATURE_COLS:
            v = geom[col]
            mu = scale.get(col, 0.0)
            sd = std.get(col, 1.0) or 1.0
            feats.append((v - mu) / sd)
        logit = bias + sum(wi * fi for wi, fi in zip(w, feats))
        prob = 1.0 / (1.0 + np.exp(-logit))
        rule = geom["e_rule_score"]
        score = 0.5 * rule + prob
        edges_scored.append((score, idx[ci], idx[cj]))

    edges_scored.sort(reverse=True)
    uf = _UnionFind(len(cids))
    for score, a, b in edges_scored:
        if score < theta_merge:
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
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def border_core_to_instances_mode(
    semantic: np.ndarray,
    core_label: int,
    border_label: int,
    target_range: tuple[int, int],
    voxel_volume_mm3: float,
    min_volume_mm3: float = 500.0,
    mode: str = "legacy",
    merge_weights: dict | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    theta_merge: float = 0.35,
    tau_intact: float = 0.08,
) -> np.ndarray:
    from scripts.pivot_form_postproc import border_core_to_instances

    if mode == "form_intact":
        cfg = FormConfig(
            min_volume_mm3=min_volume_mm3,
            tau_intact=tau_intact,
            theta_merge=10.0,
        )
        return border_core_to_instances(
            semantic,
            core_label,
            border_label,
            target_range,
            voxel_volume_mm3,
            min_volume_mm3,
            form_cfg=cfg,
            spacing_zyx=spacing_zyx,
        )

    if mode == "legacy":
        return border_core_to_instances(
            semantic,
            core_label,
            border_label,
            target_range,
            voxel_volume_mm3,
            min_volume_mm3,
            form_cfg=None,
            spacing_zyx=spacing_zyx,
        )

    out = np.zeros_like(semantic, dtype=np.uint16)
    core_mask = semantic == core_label
    border_mask = semantic == border_label
    if not core_mask.any():
        return out

    labels, n = cc_label(core_mask, structure=STRUCT26)
    if n == 0:
        return out

    if mode == "keep_all":
        fragments = fragments_keep_all(core_mask, voxel_volume_mm3, min_volume_mm3)
    elif mode == "hybrid":
        fragments = fragments_hybrid_legacy_keep(
            core_mask, border_mask, voxel_volume_mm3, spacing_zyx, min_volume_mm3
        )
    elif mode == "learned":
        if merge_weights is None:
            raise ValueError("learned mode requires merge_weights")
        fragments = fragments_learned_merge(
            core_mask,
            border_mask,
            voxel_volume_mm3,
            spacing_zyx,
            min_volume_mm3,
            merge_weights,
            theta_merge=theta_merge,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    lo, hi = target_range
    max_slots = hi - lo + 1
    fragments = fragments[:max_slots]
    for slot, (cid, _) in enumerate(fragments):
        out[labels == cid] = lo + slot
    return expand_border_into_instances(out, border_mask)


def assemble_pelvic_instances_mode(
    semantic_7class: np.ndarray,
    voxel_volume_mm3: float,
    mode: str = "legacy",
    merge_weights: dict | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_volume_mm3: float = 500.0,
    theta_merge: float = 0.35,
    tau_intact: float = 0.08,
) -> np.ndarray:
    from scripts.measure_border_core_recovery import ANATOMY

    out = np.zeros_like(semantic_7class, dtype=np.uint16)
    for name, (core_l, border_l) in {
        "sacrum": (1, 2),
        "leftHip": (3, 4),
        "rightHip": (5, 6),
    }.items():
        lo, hi = ANATOMY[name][2]
        ids = border_core_to_instances_mode(
            semantic_7class,
            core_l,
            border_l,
            (lo, hi),
            voxel_volume_mm3,
            min_volume_mm3,
            mode=mode,
            merge_weights=merge_weights,
            spacing_zyx=spacing_zyx,
            theta_merge=theta_merge,
            tau_intact=tau_intact,
        )
        out += ids
    return out


def load_merge_weights(path: Path | None = None) -> dict:
    p = path or DEFAULT_MODEL
    return json.loads(p.read_text())
