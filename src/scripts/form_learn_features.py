"""Seam / uncertainty features on core-core contact voxels (§4A.8.2)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label

STRUCT26 = generate_binary_structure(3, 3)

# Border class ids per anatomy core (D015 7-class)
CORE_TO_BORDER = {1: 2, 3: 4, 5: 6}


def contact_mask_between_cores(
    labels: np.ndarray,
    ci: int,
    cj: int,
    border_mask: np.ndarray,
) -> np.ndarray:
    mi, mj = labels == ci, labels == cj
    band = binary_dilation(mi | mj, structure=STRUCT26, iterations=1) & border_mask
    return band


def seam_features_from_probs(
    probs: np.ndarray,
    contact: np.ndarray,
    border_cls: int,
    core_cls: int,
) -> dict[str, float]:
    """Pool softmax stats on contact voxels. probs: (C, Z, Y, X)."""
    if not contact.any():
        return {
            "e_border_prob_mean": 0.0,
            "e_border_prob_max": 0.0,
            "e_core_prob_mean": 0.0,
            "e_border_entropy_mean": 0.0,
            "e_sigma_proxy_mean": 0.0,
        }
    n = int(contact.sum())
    flat = contact.ravel()
    p_all = probs[:, contact].reshape(probs.shape[0], -1)  # (C, N)
    p_border = p_all[border_cls]
    p_core = p_all[core_cls]
    eps = 1e-8
    ent = -(p_all * np.log(p_all + eps)).sum(axis=0)
    # σ proxy: high when border prob low but spread across classes (uncertain seam)
    sigma_proxy = ent / max(np.log(probs.shape[0]), 1.0)
    return {
        "e_border_prob_mean": float(p_border.mean()),
        "e_border_prob_max": float(p_border.max()),
        "e_core_prob_mean": float(p_core.mean()),
        "e_border_entropy_mean": float(ent.mean()),
        "e_sigma_proxy_mean": float(sigma_proxy.mean()),
    }


def seam_features_from_semantic_hard(
    sem: np.ndarray,
    contact: np.ndarray,
    border_cls: int,
) -> dict[str, float]:
    """Fallback when no probability volume — hard border mask as prob 0/1."""
    if not contact.any():
        return {
            "e_border_prob_mean": 0.0,
            "e_border_prob_max": 0.0,
            "e_core_prob_mean": 0.0,
            "e_border_entropy_mean": 0.0,
            "e_sigma_proxy_mean": 0.0,
        }
    border = (sem == border_cls)[contact]
    return {
        "e_border_prob_mean": float(border.mean()),
        "e_border_prob_max": float(border.max()),
        "e_core_prob_mean": 0.0,
        "e_border_entropy_mean": 0.0,
        "e_sigma_proxy_mean": 1.0 - float(border.mean()),
    }


SEAM_FEATURE_COLS = (
    "e_border_prob_mean",
    "e_border_prob_max",
    "e_core_prob_mean",
    "e_border_entropy_mean",
    "e_sigma_proxy_mean",
)
