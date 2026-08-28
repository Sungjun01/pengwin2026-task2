"""Shared edge graph utilities for Tier B Form-Learn (§4A.8.2b)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label

from scripts.form_learn_features import (
    SEAM_FEATURE_COLS,
    contact_mask_between_cores,
    seam_features_from_probs,
    seam_features_from_semantic_hard,
)
from scripts.measure_border_core_recovery import ANATOMY

STRUCT26 = generate_binary_structure(3, 3)
OVERLAP_FRAC = 0.8

GEOM_FEATURE_COLS = (
    "e_vol_i_mm3",
    "e_vol_j_mm3",
    "e_vol_ratio",
    "e_contact_voxels",
    "e_contact_mm2",
    "e_centroid_dist_mm",
    "e_border_frac_ij",
    "e_rule_score",
)
EDGE_FEATURE_COLS = GEOM_FEATURE_COLS + SEAM_FEATURE_COLS


def cc_kept(
    core_mask: np.ndarray, voxel_mm3: float, min_mm3: float
) -> tuple[np.ndarray, list[int], np.ndarray]:
    if not core_mask.any():
        return np.zeros_like(core_mask, dtype=np.int32), [], np.zeros(0, dtype=np.int64)
    labels, n = cc_label(core_mask, structure=STRUCT26)
    min_vox = int(np.ceil(min_mm3 / voxel_mm3))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]
    kept = [int(i + 1) for i, s in enumerate(sizes) if s >= min_vox]
    return labels, kept, sizes


def assign_core_to_gt(
    core_mask: np.ndarray,
    ginst: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[int | None, bool]:
    n_core = int(core_mask.sum())
    if n_core == 0:
        return None, True
    best_fid, best_ov = None, 0
    for fid in range(lo, hi + 1):
        ov = int((core_mask & (ginst == fid)).sum())
        if ov > best_ov:
            best_ov, best_fid = ov, fid
    if best_fid is None or best_ov == 0:
        return None, True
    if best_ov < OVERLAP_FRAC * n_core:
        return None, True
    return best_fid, False


def contact_edge_pairs(
    labels: np.ndarray, kept: list[int], border_mask: np.ndarray, dilation_iters: int = 2
) -> list[tuple[int, int]]:
    if len(kept) < 2:
        return []
    border_near = binary_dilation(border_mask, structure=STRUCT26, iterations=dilation_iters)
    pairs: list[tuple[int, int]] = []
    for i, ci in enumerate(kept):
        mi = labels == ci
        for cj in kept[i + 1 :]:
            if np.any(border_near & mi) and np.any(border_near & (labels == cj)):
                pairs.append((ci, cj))
    return pairs


def edge_geometry(
    labels: np.ndarray,
    ci: int,
    cj: int,
    border_mask: np.ndarray,
    voxel_mm3: float,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    mi, mj = labels == ci, labels == cj
    vi, vj = int(mi.sum()), int(mj.sum())
    contact = binary_dilation(mi | mj, structure=STRUCT26, iterations=1) & border_mask
    n_contact = int(contact.sum())
    zi, yi, xi = np.where(mi)
    zj, yj, xj = np.where(mj)
    ci_mm = np.array(
        [zi.mean() * spacing_zyx[0], yi.mean() * spacing_zyx[1], xi.mean() * spacing_zyx[2]]
    )
    cj_mm = np.array(
        [zj.mean() * spacing_zyx[0], yj.mean() * spacing_zyx[1], xj.mean() * spacing_zyx[2]]
    )
    dist_mm = float(np.linalg.norm(ci_mm - cj_mm))
    bridge = n_contact / max(min(vi, vj), 1)
    size_term = float(np.exp(-abs(np.log(max(vi, 1)) - np.log(max(vj, 1)))))
    dist_term = dist_mm / 80.0
    rule_score = 1.0 * bridge + 0.5 * size_term - 0.3 * dist_term
    return {
        "e_vol_i_mm3": vi * voxel_mm3,
        "e_vol_j_mm3": vj * voxel_mm3,
        "e_vol_ratio": vi / max(vj, 1),
        "e_contact_voxels": float(n_contact),
        "e_contact_mm2": n_contact * (voxel_mm3 ** (2 / 3)),
        "e_centroid_dist_mm": dist_mm,
        "e_border_frac_ij": n_contact / max(n_contact + vi + vj, 1),
        "e_rule_score": rule_score,
    }


def build_case_edge_rows(
    case_id: str,
    sem: np.ndarray,
    ginst: np.ndarray,
    voxel_mm3: float,
    spacing_zyx: tuple[float, float, float],
    min_volume_mm3: float,
    probs: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []
    cid = f"PENGWIN_{case_id}"
    for anatomy, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
        core = sem == core_cls
        border = sem == border_cls
        labels, kept, _ = cc_kept(core, voxel_mm3, min_volume_mm3)
        if len(kept) < 2:
            continue

        gt_map: dict[int, tuple[int | None, bool]] = {}
        for c in kept:
            gt_map[c] = assign_core_to_gt(labels == c, ginst, lo, hi)

        for ci, cj in contact_edge_pairs(labels, kept, border):
            gi, abst_i = gt_map[ci]
            gj, abst_j = gt_map[cj]
            if abst_i or abst_j or gi is None or gj is None:
                y_merge = None
                abstain = True
            elif gi == gj:
                y_merge = 1
                abstain = False
            else:
                y_merge = 0
                abstain = False

            geom = edge_geometry(labels, ci, cj, border, voxel_mm3, spacing_zyx)
            contact = contact_mask_between_cores(labels, ci, cj, border)
            if probs is not None and probs.shape[0] >= 7:
                seam = seam_features_from_probs(probs, contact, border_cls, core_cls)
            else:
                seam = seam_features_from_semantic_hard(sem, contact, border_cls)
            rows.append(
                {
                    "case_id": cid,
                    "anatomy": anatomy,
                    "core_i": ci,
                    "core_j": cj,
                    "y_merge": y_merge,
                    "abstain_flag": abstain,
                    "gt_id_i": gi,
                    "gt_id_j": gj,
                    "n_cores": len(kept),
                    **geom,
                    **seam,
                }
            )
    return rows
