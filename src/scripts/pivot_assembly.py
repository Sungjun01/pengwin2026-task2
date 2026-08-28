"""PIVOT-Assembly: conservative graph merge for fragment instances.

This module keeps the v10 segmentation/model outputs fixed and only merges
small, nearby fragments when there is strong geometric evidence that they are
decode slivers rather than independent fracture fragments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass


STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class AssemblyConfig:
    max_sliver_volume_mm3: float = 2000.0
    min_volume_ratio: float = 5.0
    dilation_iters: int = 1
    max_centroid_distance_mm: float = 30.0


@dataclass(frozen=True)
class CorticalContinuityConfig:
    merge_score_threshold: float = 0.65
    cortical_hu_threshold: float = 300.0
    fracture_gap_hu_threshold: float = 120.0
    interface_dilation_iters: int = 1


@dataclass(frozen=True)
class AssemblyEdge:
    small_id: int
    large_id: int
    small_volume_mm3: float
    large_volume_mm3: float
    volume_ratio: float
    centroid_distance_mm: float
    contact_voxels: int
    contact_ratio: float


def anatomy_key(instance_id: int) -> str | None:
    if 1 <= instance_id <= 50:
        return "sacrum"
    if 51 <= instance_id <= 100:
        return "leftHip"
    if 101 <= instance_id <= 150:
        return "rightHip"
    if 151 <= instance_id <= 200:
        return "femur"
    return None


def _centroid_mm(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    cz, cy, cx = center_of_mass(mask)
    sz, sy, sx = spacing_zyx
    return np.array([cz * sz, cy * sy, cx * sx], dtype=np.float64)


def extract_assembly_edges(
    instance_arr: np.ndarray,
    voxel_volume_mm3: float,
    cfg: AssemblyConfig | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[AssemblyEdge]:
    """Return conservative small-to-large merge candidates."""
    cfg = cfg or AssemblyConfig()
    ids = sorted(int(x) for x in np.unique(instance_arr) if x != 0)
    masks = {iid: instance_arr == iid for iid in ids}
    volumes = {iid: float(masks[iid].sum()) * voxel_volume_mm3 for iid in ids}
    centroids = {iid: _centroid_mm(masks[iid], spacing_zyx) for iid in ids}
    dilated = {
        iid: binary_dilation(masks[iid], structure=STRUCT26, iterations=cfg.dilation_iters)
        for iid in ids
    }

    edges: list[AssemblyEdge] = []
    for small_id in ids:
        small_anat = anatomy_key(small_id)
        small_vol = volumes[small_id]
        if small_anat is None or small_vol > cfg.max_sliver_volume_mm3:
            continue

        small_surface = max(int(dilated[small_id].sum() - masks[small_id].sum()), 1)
        for large_id in ids:
            if large_id == small_id or anatomy_key(large_id) != small_anat:
                continue
            large_vol = volumes[large_id]
            if large_vol <= small_vol:
                continue
            volume_ratio = large_vol / max(small_vol, 1e-6)
            if volume_ratio < cfg.min_volume_ratio:
                continue

            contact = int((dilated[small_id] & masks[large_id]).sum())
            if contact == 0:
                continue
            dist = float(np.linalg.norm(centroids[small_id] - centroids[large_id]))
            if dist > cfg.max_centroid_distance_mm:
                continue

            edges.append(
                AssemblyEdge(
                    small_id=small_id,
                    large_id=large_id,
                    small_volume_mm3=small_vol,
                    large_volume_mm3=large_vol,
                    volume_ratio=volume_ratio,
                    centroid_distance_mm=dist,
                    contact_voxels=contact,
                    contact_ratio=contact / small_surface,
                )
            )
    return edges


def merge_sliver_fragments(
    instance_arr: np.ndarray,
    voxel_volume_mm3: float,
    cfg: AssemblyConfig | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Merge each sliver into its strongest same-anatomy large-fragment edge."""
    out = instance_arr.copy()
    edges = extract_assembly_edges(out, voxel_volume_mm3, cfg, spacing_zyx)
    best_by_small: dict[int, AssemblyEdge] = {}
    for edge in edges:
        prev = best_by_small.get(edge.small_id)
        if prev is None or (edge.contact_voxels, edge.volume_ratio) > (
            prev.contact_voxels,
            prev.volume_ratio,
        ):
            best_by_small[edge.small_id] = edge

    for small_id, edge in best_by_small.items():
        out[out == small_id] = edge.large_id
    return out


def cortical_continuity_score(
    small_mask: np.ndarray,
    large_mask: np.ndarray,
    ct_hu: np.ndarray,
    cfg: CorticalContinuityConfig | None = None,
) -> float:
    """Score whether the CT around a candidate merge looks like continuous cortex.

    High score means a small predicted sliver is embedded in high-HU continuous
    bone and is safe to merge. Low score means the interface contains low-HU
    fracture-gap evidence, so the v18 split should be preserved.
    """
    cfg = cfg or CorticalContinuityConfig()
    if ct_hu.shape != small_mask.shape:
        raise ValueError(f"ct_hu shape {ct_hu.shape} does not match mask shape {small_mask.shape}")
    if not small_mask.any() or not large_mask.any():
        return 0.0

    small_band = binary_dilation(
        small_mask, structure=STRUCT26, iterations=cfg.interface_dilation_iters
    ) & large_mask
    large_band = binary_dilation(
        large_mask, structure=STRUCT26, iterations=cfg.interface_dilation_iters
    ) & small_mask
    interface = small_band | large_band
    if not interface.any():
        return 0.0

    vals = ct_hu[interface].astype(np.float32, copy=False)
    high_frac = float(np.mean(vals >= cfg.cortical_hu_threshold))
    low_frac = float(np.mean(vals <= cfg.fracture_gap_hu_threshold))
    return max(0.0, min(1.0, high_frac * (1.0 - low_frac)))


def merge_sliver_fragments_with_ct_evidence(
    instance_arr: np.ndarray,
    voxel_volume_mm3: float,
    ct_hu: np.ndarray | None,
    cfg: AssemblyConfig | None = None,
    evidence_cfg: CorticalContinuityConfig | None = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Merge slivers only when local CT evidence supports cortical continuity."""
    if ct_hu is None:
        return merge_sliver_fragments(instance_arr, voxel_volume_mm3, cfg, spacing_zyx)
    if ct_hu.shape != instance_arr.shape:
        raise ValueError(f"ct_hu shape {ct_hu.shape} does not match instance shape {instance_arr.shape}")

    evidence_cfg = evidence_cfg or CorticalContinuityConfig()
    out = instance_arr.copy()
    edges = extract_assembly_edges(out, voxel_volume_mm3, cfg, spacing_zyx)
    best_by_small: dict[int, tuple[AssemblyEdge, float]] = {}
    for edge in edges:
        score = cortical_continuity_score(
            out == edge.small_id,
            out == edge.large_id,
            ct_hu,
            evidence_cfg,
        )
        if score < evidence_cfg.merge_score_threshold:
            continue
        prev = best_by_small.get(edge.small_id)
        if prev is None or (edge.contact_voxels, score, edge.volume_ratio) > (
            prev[0].contact_voxels,
            prev[1],
            prev[0].volume_ratio,
        ):
            best_by_small[edge.small_id] = (edge, score)

    for small_id, (edge, _) in best_by_small.items():
        out[out == small_id] = edge.large_id
    return out


def merge_femur_chunks_by_spatial_groups(
    instance_arr: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_chunks: int = 3,
    min_group_gap_mm: float = 20.0,
    target_range: tuple[int, int] = (151, 200),
) -> np.ndarray:
    """Group over-split femur watershed chunks into two spatial femur groups.

    This is intentionally conservative: it only acts when there are at least
    `min_chunks` femur instances and their centroids show a clear left/right
    separation along physical x. It avoids the v16 failure mode of always
    collapsing femur output to one fragment.
    """
    out = instance_arr.copy()
    lo, hi = target_range
    femur_ids = sorted(int(x) for x in np.unique(instance_arr) if lo <= int(x) <= hi)
    if len(femur_ids) < min_chunks:
        return out

    centroids_x: list[tuple[int, float]] = []
    sz, sy, sx = spacing_zyx
    for iid in femur_ids:
        mask = instance_arr == iid
        if not mask.any():
            continue
        _, _, cx = center_of_mass(mask)
        centroids_x.append((iid, float(cx) * sx))
    if len(centroids_x) < min_chunks:
        return out

    centroids_x.sort(key=lambda x: x[1])
    gaps = [
        (centroids_x[i + 1][1] - centroids_x[i][1], i)
        for i in range(len(centroids_x) - 1)
    ]
    max_gap, split_idx = max(gaps, key=lambda x: x[0])
    if max_gap < min_group_gap_mm:
        return out

    groups = [centroids_x[: split_idx + 1], centroids_x[split_idx + 1 :]]
    for group_idx, group in enumerate(groups):
        new_id = lo + group_idx
        for old_id, _ in group:
            out[instance_arr == old_id] = new_id
    return out


def _split_bilateral_x(comp: np.ndarray, x_axis: int = 2) -> list[np.ndarray]:
    """컴포넌트가 *붙은 좌/우 femur*(x축 bimodal + 얇은 목)면 valley에서 2개로 분리,
    단일 femur(unimodal)면 [comp] 그대로. 중앙 걸친 단일 femur 오분할 방지 가드 포함."""
    other = tuple(a for a in range(comp.ndim) if a != x_axis)
    xs = comp.sum(axis=other).astype(np.float64)  # x 슬라이스별 mass
    nz = np.nonzero(xs)[0]
    if len(nz) < 8:
        return [comp]
    x0, x1 = int(nz[0]), int(nz[-1])
    if x1 - x0 < 16:
        return [comp]
    c0, c1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
    if c1 <= c0:
        return [comp]
    valley = c0 + int(np.argmin(xs[c0:c1 + 1]))
    left, right, total = xs[:valley].sum(), xs[valley:].sum(), xs.sum()
    central_max = float(xs[c0:c1 + 1].max())
    # 양측 다 substantial(>20%) + valley가 충분히 낮음(<25% 중앙최대 = 진짜 좌/우 분리)
    if left > 0.2 * total and right > 0.2 * total and xs[valley] < 0.25 * central_max:
        sl_left = [slice(None)] * comp.ndim; sl_left[x_axis] = slice(valley, None)
        sl_right = [slice(None)] * comp.ndim; sl_right[x_axis] = slice(0, valley)
        lm = comp.copy(); lm[tuple(sl_left)] = False
        rm = comp.copy(); rm[tuple(sl_right)] = False
        return [lm, rm]
    return [comp]


def hybrid_femur_decode_global_cc(
    d004_binary: np.ndarray,
    d013_core: np.ndarray,
    d013_border: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_range: tuple[int, int] = (151, 200),
    min_volume_mm3: float = 500.0,
    min_piece_mm3: float = 500.0,
) -> np.ndarray:
    """v27-True-v2 하이브리드 femur 디코더 (over-split + bilateral under-seg 동시 봉인).

    1) D004 binary 글로벌 26-conn CC + *x-valley 좌/우 분리*(붙은 양측도 격리, bilateral merge 차단).
    2) 각 (측)덩어리 *내부에서만* D013 core CC →
       - ≤1 = 온전한 뼈 → 1 인스턴스(under-seg 구출, watershed 과분할 회피).
       - ≥2 = 골절 → 덩어리 안에서만 watershed 분할.
    """
    from scipy.ndimage import label as _label, distance_transform_edt as _edt
    from skimage.segmentation import watershed as _ws

    lo, hi = target_range
    out = np.zeros(d004_binary.shape, dtype=np.uint16)
    glabels, ng = _label(d004_binary > 0, structure=STRUCT26)
    comps = sorted(((int((glabels == g).sum()), g) for g in range(1, ng + 1)), reverse=True)
    nid = lo
    for sz, g in comps:
        if sz * voxel_volume_mm3 < min_volume_mm3 or nid > hi:
            continue
        comp = glabels == g
        for sub in _split_bilateral_x(comp):           # ← 붙은 좌/우 분리
            if nid > hi or int(sub.sum()) * voxel_volume_mm3 < min_volume_mm3:
                continue
            local_core = d013_core & sub
            llab, nl = _label(local_core, structure=STRUCT26)
            pieces = [l for l in range(1, nl + 1)
                      if int((llab == l).sum()) * voxel_volume_mm3 >= min_piece_mm3]
            if len(pieces) <= 1:
                out[sub] = nid
                nid += 1
            else:
                markers = np.zeros(sub.shape, dtype=np.int32)
                for slot, l in enumerate(pieces, start=1):
                    markers[llab == l] = slot
                elev = _edt(markers == 0, sampling=spacing_zyx)
                ws = _ws(elev, markers=markers, mask=sub)
                for slot in range(1, len(pieces) + 1):
                    if nid > hi:
                        break
                    out[ws == slot] = nid
                    nid += 1
    return out


def monotone_safe_femur_decode(
    d004_binary: np.ndarray,
    d013_core: np.ndarray,
    d013_border: np.ndarray,
    voxel_volume_mm3: float,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    target_range: tuple[int, int] = (151, 200),
    min_volume_mm3: float = 500.0,
    min_peak_distance: int = 20,
    core_overlap_frac: float = 0.10,
    min_core_mm3: float = 200.0,
) -> np.ndarray:
    """v28 Monotone-Safe femur 디코더 — watershed FLOOR + d013 양성증거 병합.

    `feedback_monotone_safe_decode` 원칙 구현 (v27 참사 차단):
      1) 기하 바닥: d004 binary를 watershed로 *과분할*. 전위 골절·얇은 목은
         데이터 일반화 여부와 무관하게 무조건 1차 분리(알고리즘 하한선).
      2) DL은 양성증거(병합)만: 두 watershed chunk가 *같은 d013 core 연결성분*에
         substantial하게 덮일 때만 병합("같은 뼈"라는 양성신호). DL은 기하 분할을
         지우거나 강제봉합하지 못함 — *병합만* 가능.
      3) 침묵 = 무조치: core가 안 덮은 chunk(d013 recall-miss/침묵)는 *그대로* 자기
         인스턴스로 유지 → watershed 분할 보존. d013가 안 본 전위 조각을 침묵으로
         지나쳐도 갈라진 채 남음(JAM-UNDER 핵폭탄 차단).
      4) 하한 보장: d013가 전부 침묵하면 결과 = 순수 watershed floor(≈v18 femur).
         다운사이드가 watershed로 *묶임*.

    v27 global-cc와 결정적 차이: v27은 "1 blob=1 인스턴스"에서 *출발*해 d013이
    분할을 *허락*해야만 쪼갰다(침묵→봉합=참사). v28은 watershed 과분할에서 출발해
    d013이 *양성증거를 줄 때만* 병합한다(침묵→분할유지=안전).
    """
    from collections import Counter
    from scipy.ndimage import distance_transform_edt, label as _label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as _ws

    lo, hi = target_range
    out = np.zeros(d004_binary.shape, dtype=np.uint16)
    mask = d004_binary > 0
    if not mask.any():
        return out

    # --- 1) 기하 바닥: watershed 과분할 (v18 floor와 동일 로직) ---
    distance = distance_transform_edt(mask)
    peaks = peak_local_max(
        distance, min_distance=min_peak_distance,
        labels=mask.astype(np.int32), exclude_border=False,
    )
    if len(peaks) == 0:
        ws_labels, K = _label(mask, structure=STRUCT26)
    else:
        markers = np.zeros(mask.shape, dtype=np.int32)
        for idx, coord in enumerate(peaks, start=1):
            markers[tuple(coord)] = idx
        ws_labels = _ws(-distance, markers=markers, mask=mask)
        K = int(ws_labels.max())
    if K == 0:
        return out

    # --- 2) d013 core 연결성분 (병합의 양성증거 단위) ---
    core_cc, ncore = _label(d013_core & mask, structure=STRUCT26)

    # 각 watershed chunk → 지배적 core CC (substantial 겹침일 때만), 아니면 0(침묵)
    chunk_to_core: dict[int, int] = {}
    chunk_size: dict[int, int] = {}
    for k in range(1, K + 1):
        chunk = ws_labels == k
        sz = int(chunk.sum())
        chunk_size[k] = sz
        if sz == 0:
            chunk_to_core[k] = 0
            continue
        ov = core_cc[chunk]
        ov = ov[ov > 0]
        if ov.size == 0:
            chunk_to_core[k] = 0           # 침묵 → 분할 유지
            continue
        dom, dom_cnt = Counter(ov.tolist()).most_common(1)[0]
        # substantial: chunk의 10%↑ 또는 절대 200mm³↑ core가 있어야 "core-backed"
        if dom_cnt >= core_overlap_frac * sz or dom_cnt * voxel_volume_mm3 >= min_core_mm3:
            chunk_to_core[k] = int(dom)
        else:
            chunk_to_core[k] = 0           # core 미미 → 침묵 취급, 분할 유지

    # --- 3) union-find: 같은 core CC를 공유하는 chunk만 병합 (양성증거) ---
    parent = {k: k for k in range(1, K + 1)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_core: dict[int, list[int]] = {}
    for k, c in chunk_to_core.items():
        if c > 0:
            by_core.setdefault(c, []).append(k)
    for c, ks in by_core.items():
        for k in ks[1:]:
            union(ks[0], k)   # 같은 core CC = 같은 뼈 → 병합 / 침묵 chunk는 singleton

    # --- 4) 그룹별 인스턴스 ID 부여 (size 내림차순, min_volume 컷, target_range 캡) ---
    groups: dict[int, list[int]] = {}
    for k in range(1, K + 1):
        groups.setdefault(find(k), []).append(k)
    sized = sorted(
        ((sum(chunk_size[k] for k in ks), ks) for ks in groups.values()),
        key=lambda x: x[0], reverse=True,
    )
    nid = lo
    for tot, ks in sized:
        if tot * voxel_volume_mm3 < min_volume_mm3 or nid > hi:
            continue
        for k in ks:
            out[ws_labels == k] = nid
        nid += 1
    return out
