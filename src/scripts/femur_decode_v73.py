"""Task2 대퇴 디코드 v73 — 씨앗을 클릭에서 얻고, 클릭이 약할 때만 추측한다.

WHY. Task2 대퇴가 리더보드 0.8883 인데 **클릭 없는** Task1 v62 가 0.9069 였다.
클릭은 조각 개수와 위치를 알려주는 정보인데 자동보다 못하다는 건 파이프라인이
낡았다는 뜻이다. 실제로 Task2 의 대퇴 auto-seg 를 학습 170 케이스에서 재보면
**0.7241** 이고, 클릭 후처리가 그걸 0.9081 까지 끌어올리고 있었다 — 밑바닥이 얼마나
나쁜지 클릭이 가려주고 있었다.

바꾼 것: 대퇴 전경을 D004 대신 **D013 argmax** 로 잡고(이미 seam 용으로 D013 을
돌리면서 argmax 를 버리고 있었다 — 추론 추가 없음), 씨앗을 클릭에서 직접 얻는다.

측정 (학습 femur 170케이스, Task2 가 실제로 쓰는 FemurSnap 모델, 케이스단위 OOF):

    클릭 전략                        B 클릭씨앗   C 클릭+hmax+병합   A 병합만
    center_of_mass                   0.9796       0.9559
    euclidean_distance_transform     0.9838       0.9574
    uniformly_sampled                0.9611       0.9573
    boundary_internal_margin         0.8874       0.9503          0.9612
    ------------------------------------------------------------------
    전부 B = 0.9530 · 전부 C = 0.9552 · **B + 경계는 A = 0.9714**

가장자리 클릭만 유독 약한 이유는 기전으로 설명된다: 씨앗이 능선 가까이 있으면 물이
옆 조각으로 넘어가 두 조각이 하나로 나온다(recall 0.8387). 그래서 그 전략에서만
클릭을 씨앗으로 쓰지 않고 h-maxima 과분할 + 학습된 병합으로 간다.

**전략은 입력 JSON 의 name 키에 적혀 있다** — GT 가 아니라 주어지는 정보이므로
이 분기는 정당하다.

비교: Task2 현재 0.9081, Task2 1등 대퇴 0.9572.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

ST26 = np.ones((3, 3, 3), bool)
FEATS = ["contact_mm2", "face_mean", "face_med", "face_p10", "face_p90", "face_max",
         "face_frac_hi", "vol_min_mm3", "vol_max_mm3", "vol_ratio",
         "depth_min", "depth_max", "bp_in_min", "bp_in_max", "contact_over_vol"]


def load_model(path=None):
    p = path or os.environ.get("PIVOT_FEMUR_MERGE_MODEL", "")
    if not p or not Path(p).exists():
        return None
    return json.loads(Path(p).read_text())


def is_boundary_strategy(click_json_path) -> bool:
    """클릭 전략이 '가장자리'인가. 입력 JSON 의 name 키에 적혀 있다."""
    try:
        name = json.loads(Path(click_json_path).read_text()).get("name", "")
    except Exception:
        return False
    n = name.lower()
    return "boundary" in n or "internal margin" in n


def _pair_rows(seg, bp, vox, spacing_zyx):
    """인접 조각 쌍마다 특징. 학습 때 쓴 정의와 같아야 한다."""
    ids = [int(v) for v in np.unique(seg) if v > 0]
    if len(ids) < 2:
        return [], []
    size = {i: int((seg == i).sum()) for i in ids}
    edt = distance_transform_edt(seg > 0, sampling=spacing_zyx)
    depth = {i: float(edt[seg == i].max()) for i in ids}
    bp_in = {i: float(bp[seg == i].mean()) for i in ids}
    agg = {}
    for ax in range(3):
        a = np.take(seg, np.arange(0, seg.shape[ax] - 1), axis=ax)
        b = np.take(seg, np.arange(1, seg.shape[ax]), axis=ax)
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        pa, pb = a[m], b[m]
        face = np.maximum(np.take(bp, np.arange(0, bp.shape[ax] - 1), axis=ax)[m],
                          np.take(bp, np.arange(1, bp.shape[ax]), axis=ax)[m])
        key = np.minimum(pa, pb) * 100000 + np.maximum(pa, pb)
        for k in np.unique(key):
            i, j = int(k // 100000), int(k % 100000)
            f = face[key == k]
            agg[(i, j)] = np.concatenate([agg[(i, j)], f]) if (i, j) in agg else f
    va = float(np.prod(spacing_zyx) ** (2 / 3))
    pairs, rows = [], []
    for (i, j), f in agg.items():
        n = len(f)
        vi, vj = size[i] * vox, size[j] * vox
        vlo, vhi = min(vi, vj), max(vi, vj)
        rows.append({
            "contact_mm2": n * va, "face_mean": float(f.mean()),
            "face_med": float(np.median(f)), "face_p10": float(np.percentile(f, 10)),
            "face_p90": float(np.percentile(f, 90)), "face_max": float(f.max()),
            "face_frac_hi": float((f > 0.5).mean()),
            "vol_min_mm3": vlo, "vol_max_mm3": vhi, "vol_ratio": vlo / max(vhi, 1e-9),
            "depth_min": min(depth[i], depth[j]), "depth_max": max(depth[i], depth[j]),
            "bp_in_min": min(bp_in[i], bp_in[j]), "bp_in_max": max(bp_in[i], bp_in[j]),
            "contact_over_vol": (n * va) / max(vlo ** (2 / 3), 1e-6),
        })
        pairs.append((i, j))
    return pairs, rows


def _merge(seg, pairs, rows, model):
    logf = set(model["log_feats"])
    X = np.array([[np.log1p(max(r[k], 0.0)) if k in logf else r[k]
                   for k in model["feats"]] for r in rows], np.float64)
    Z = np.hstack([(X - np.array(model["mu"])) / np.array(model["sd"]),
                   np.ones((len(X), 1))])
    p = 1.0 / (1.0 + np.exp(-Z @ np.array(model["w"])))
    n = int(seg.max())
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), pv in zip(pairs, p):
        if pv >= model["thr"]:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    return np.array([find(k) for k in range(n + 1)], np.int32)


def femur_decode(mask, bp, click_points, spacing_zyx, voxel_volume_mm3,
                 target_range=(151, 200), min_volume_mm3=1000.0,
                 use_clicks=True, model=None):
    """대퇴 인스턴스. use_clicks=True 면 클릭이 씨앗, False 면 h-maxima + 학습된 병합."""
    lo, hi = target_range
    out = np.zeros(mask.shape, np.uint16)
    if not mask.any():
        return out
    idx = np.where(mask)
    sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
    m2 = mask[sl]
    bp2 = np.asarray(bp[sl], dtype=np.float32)

    if use_clicks and click_points:
        markers = np.zeros(m2.shape, np.int32)
        mk = 0
        ei = None
        for pt in click_points:
            v = tuple(int(np.clip(int(round(pt[i])), 0, mask.shape[i] - 1)) for i in range(3))
            if not mask[v]:
                if ei is None:
                    _d, ei = distance_transform_edt(~mask, return_indices=True)
                v = tuple(int(ei[ax][v]) for ax in range(3))
            lv = tuple(v[i] - sl[i].start for i in range(3))
            if not all(0 <= lv[i] < m2.shape[i] for i in range(3)):
                continue
            mk += 1
            seed = np.zeros(m2.shape, bool)
            seed[lv] = True
            markers[binary_dilation(seed, ST26) & m2] = mk
        if mk == 0:
            return out
        seg2 = watershed(bp2, markers=markers, mask=m2).astype(np.int32)
        lut = None
    else:
        inner = m2 & (bp2 < model["tau"])
        if not inner.any():
            return out
        dist = distance_transform_edt(inner, sampling=spacing_zyx)
        markers, mk = cc_label(h_maxima(dist, model["h"]) & inner, structure=ST26)
        if mk == 0:
            return out
        seg2 = watershed(bp2, markers=markers.astype(np.int32), mask=m2).astype(np.int32)
        seg_full = np.zeros(mask.shape, np.int32)
        seg_full[sl] = seg2
        pairs, rows = _pair_rows(seg_full, bp, voxel_volume_mm3, spacing_zyx)
        lut = _merge(seg_full, pairs, rows, model) if rows else None

    if lut is not None:
        seg2 = lut[seg2]
    keep = []
    for v in np.unique(seg2):
        if v <= 0:
            continue
        m = seg2 == v
        vol = int(m.sum()) * voxel_volume_mm3
        if vol >= min_volume_mm3:
            keep.append((vol, m))
    keep.sort(key=lambda t: -t[0])
    sub = out[sl]
    for k, (_v, m) in enumerate(keep[:hi - lo + 1]):
        sub[m] = lo + k
    return out
