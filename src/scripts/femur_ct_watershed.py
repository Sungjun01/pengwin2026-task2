"""CT-유도 femur watershed — 골절면을 *기하 thin-neck*이 아니라 *실제 균열(어두운 CT)*에 놓음.

기존 per_anatomy_watershed_instance는 elevation=-distance(거리장) → 경계가 기하 중간선(thin neck).
displaced/slant 골절이면 실제 균열면과 어긋나 hd95 오차(004: 8.67mm vs 1등 2.14).
여기선 elevation에 CT 어두움(=균열)을 ridge로 섞어 경계가 균열면을 따라가게 → hd95/dice 개선.
(사크럼 propagation_completion과 동일 원리, femur binary 버전.)
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, generate_binary_structure, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

STRUCT26 = generate_binary_structure(3, 3)


def per_anatomy_ct_watershed_instance(
    semantic_arr: np.ndarray,
    ct_arr: np.ndarray,
    anatomy_class: int,
    target_range: tuple,
    voxel_volume_mm3: float,
    min_volume_mm3: float = 500.0,
    min_peak_distance: int = 20,
    w_ct: float = 1.0,
    w_dist: float = 0.5,
    smooth: float = 1.0,
) -> np.ndarray:
    """femur watershed (CT-유도). markers는 distance peak 그대로, elevation만 CT-유도로 교체."""
    mask = (semantic_arr == anatomy_class)
    if not mask.any():
        return np.zeros_like(semantic_arr, dtype=np.uint16)
    distance = distance_transform_edt(mask)
    peak_coords = peak_local_max(distance, min_distance=min_peak_distance,
                                 labels=mask.astype(np.int32), exclude_border=False)
    if len(peak_coords) == 0:
        labels, _ = cc_label(mask, structure=STRUCT26)
    else:
        markers = np.zeros(mask.shape, dtype=np.int32)
        for idx, coord in enumerate(peak_coords, start=1):
            markers[tuple(coord)] = idx
        # CT-유도 elevation: 어두움(균열)=high(ridge), 깊은 내부=low(basin)
        cs = gaussian_filter(ct_arr.astype(np.float32), smooth)
        fv = cs[mask]
        lo, hi = np.percentile(fv, 10), np.percentile(fv, 90)
        ctn = np.clip((cs - lo) / (hi - lo + 1e-6), 0, 1)
        edtn = distance / (distance.max() + 1e-6)
        elev = w_ct * (1.0 - ctn) - w_dist * edtn
        labels = watershed(elev, markers=markers, mask=mask)
    n = int(labels.max())
    if n == 0:
        return np.zeros_like(semantic_arr, dtype=np.uint16)
    sizes = [(cid, int((labels == cid).sum())) for cid in range(1, n + 1)]
    kept = [(cid, sz) for cid, sz in sizes if sz * voxel_volume_mm3 >= min_volume_mm3]
    kept.sort(key=lambda x: x[1], reverse=True)
    lo_r, hi_r = target_range
    kept = kept[: hi_r - lo_r + 1]
    out = np.zeros_like(semantic_arr, dtype=np.uint16)
    for slot, (cid, _) in enumerate(kept):
        out[labels == cid] = lo_r + slot
    return out
