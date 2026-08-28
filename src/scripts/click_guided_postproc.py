"""Task2 click-guided relabeling on top of Task1 (v19) instance predictions.

Approach (Task2 = v19 auto-seg + click postprocessing, no retraining)
---------------------------------------------------------------------
v19's `run_pipeline_v7` already emits a PENGWIN instance map (0=bg, 1-50 sacrum,
51-100 leftHip, 101-150 rightHip, 151-200 femur) in the original CT frame. The
Task2 clicks give exactly ONE seed per ground-truth fragment, so they pin down the
fragment count and identity -- precisely v19's weakest signal. We therefore RESPECT
v19's voxel boundaries and only intervene where clicks disagree:

  * v19 instance with 1 click   -> keep the whole component as that fragment.
  * v19 instance with >=2 clicks -> v19 merged two fragments: watershed-split it,
                                    seeded by the clicks.
  * v19 instance with 0 clicks   -> either a false positive or an over-split shard.
                                    Absorb it into a touching clicked fragment;
                                    drop it if isolated.

Coordinate convention (the silent-breakage trap)
------------------------------------------------
The PENGWIN clicks store each point as an INTEGER NUMPY ARRAY INDEX in (z, y, x)
order. This is proven on the organizer's training files: simulate_clicks builds
points from `argwhere` on the (z,y,x) SITK array and asserts
`label_im[z, y, x] == fragment_label` (verified 40/40 voxels on case 099). Grand
Challenge may re-serialize its point interface to a different axis order, so
`_resolve_voxel` tries BOTH the as-stored (z,y,x) order and the reversed (x,y,z)
order and keeps whichever lands in / closest to the relevant bone, defaulting to
the proven (z,y,x) on ties. This makes the parser self-correcting either way.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
)
from skimage.segmentation import watershed


ANATOMY_RANGES: dict[str, tuple[int, int]] = {
    "sacrum": (1, 50),
    "leftHip": (51, 100),
    "rightHip": (101, 150),
    "femur": (151, 200),
}

_CONN26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class Click:
    """A single parsed click. `point` is the RAW coordinate as stored in the JSON."""

    anatomy: Optional[str]
    point: tuple[float, float, float]
    name: str
    fragment_id: Optional[int] = None


def _anatomy_from_name(name: str) -> Optional[str]:
    """Infer anatomy from a click name. Returns None when the name carries none."""
    text = name.lower()
    if "sacrum" in text:
        return "sacrum"
    if "femur" in text or "femoral" in text:
        return "femur"
    if "left" in text and ("hip" in text or "iliac" in text or "ilium" in text):
        return "leftHip"
    if "right" in text and ("hip" in text or "iliac" in text or "ilium" in text):
        return "rightHip"
    return None


def _range_to_anatomy(label: int) -> Optional[str]:
    for anatomy, (lo, hi) in ANATOMY_RANGES.items():
        if lo <= label <= hi:
            return anatomy
    return None


def _points_from_click_payload(data: Any) -> list[dict[str, Any]]:
    """Accept the official single MPOI object or a list of strategy objects."""
    if isinstance(data, list):
        points: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                points.extend(item.get("points", []))
        return points
    if isinstance(data, dict):
        return list(data.get("points", []))
    return []


def parse_clicks(path_or_data: str | Path | dict[str, Any] | list[Any]) -> list[Click]:
    """Parse a peripelvic-fragment-clicks.json into Click records.

    Robust to: list-of-strategies or single-dict payloads; missing anatomy in the
    name (anatomy left None, inferred later from the voxel label); an explicit
    `fragment_id` field or a trailing index in the name; float coordinates.
    """
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data) as f:
            data = json.load(f)
    else:
        data = path_or_data

    clicks: list[Click] = []
    for point in _points_from_click_payload(data):
        coord = point.get("point")
        if coord is None or len(coord) != 3:
            continue
        name = str(point.get("name", ""))
        fragment_id = point.get("fragment_id")
        if fragment_id is None:
            match = re.search(r"(\d+)\s*$", name)
            fragment_id = int(match.group(1)) if match else None
        clicks.append(
            Click(
                anatomy=_anatomy_from_name(name),
                point=tuple(float(v) for v in coord),  # type: ignore[arg-type]
                name=name,
                fragment_id=fragment_id,
            )
        )
    return clicks


def _round_clip(coords: tuple[float, ...], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(np.clip(int(round(c)), 0, shape[i] - 1)) for i, c in enumerate(coords))  # type: ignore[return-value]


def _index_candidates(
    point: tuple[float, float, float],
    image,
    shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """All plausible (z,y,x) voxel indices for a raw click `point`.

    Grand Challenge's point interface may serialize a click as a voxel index OR as a
    physical/world coordinate (mm), in either axis order. We therefore generate, in
    preference order, the voxel interpretations (z,y,x) and (x,y,z), then — when an
    image with geometry is supplied — the world->index interpretations of the point
    read both ways. The resolver picks whichever lands in the named anatomy.
    """
    cands: list[tuple[int, int, int]] = [_round_clip(point, shape), _round_clip(point[::-1], shape)]
    if image is not None:
        for p in (point, point[::-1]):
            try:
                ix, iy, iz = image.TransformPhysicalPointToContinuousIndex(
                    (float(p[0]), float(p[1]), float(p[2]))
                )
                cands.append(_round_clip((iz, iy, ix), shape))
            except Exception:
                pass
    seen: set[tuple[int, int, int]] = set()
    uniq: list[tuple[int, int, int]] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _resolve_voxel_in_mask(
    point: tuple[float, float, float],
    shape: tuple[int, int, int],
    fg_mask: np.ndarray,
    get_edt,
    image=None,
) -> Optional[tuple[int, int, int]]:
    """Resolve a click to a (z,y,x) voxel inside `fg_mask`, robust to axis order AND
    to voxel-vs-world coordinates.

    Fast path (the norm): one candidate interpretation lands inside the anatomy, so
    membership alone decides — no distance transform. The candidate order encodes the
    preference (voxel (z,y,x) first, then (x,y,z), then world->index), so a click that
    is already a valid voxel keeps its proven reading. Only when NO candidate lands
    inside do we pay for the lazily-built EDT to snap the closest candidate onto the
    mask. Returns None if the mask is empty.
    """
    cands = _index_candidates(point, image, shape)
    for c in cands:
        if bool(fg_mask[c]):
            return c
    edt = get_edt()
    if edt is None:
        return None
    dist, inds = edt
    chosen = min(cands, key=lambda c: float(dist[c]))
    return tuple(int(inds[ax][chosen]) for ax in range(3))  # type: ignore[return-value]


def _nearest_in_mask(mask: np.ndarray, point: tuple[int, int, int]) -> tuple[int, int, int]:
    _, inds = distance_transform_edt(~mask, return_indices=True)
    return tuple(int(inds[ax][point]) for ax in range(3))  # type: ignore[return-value]


def _watershed_worker(q, elev, markers, mask) -> None:
    q.put(watershed(elev, markers=markers, mask=mask, watershed_line=True))


def _watershed_safe(elev: np.ndarray, markers: np.ndarray, mask: np.ndarray, watershed_line: bool,
                     timeout: float = None) -> np.ndarray:
    """skimage.watershed, guarded against the watershed_line=True hang bug.

    Root-caused via GPU smoke test on the real public femur validation cases (004/005):
    watershed_line=True can be >2000x slower (0.04s -> does not finish in 90s+) on a
    plateau-heavy distance field -- exactly what a long/thin bone shaft produces -- with
    unbounded RSS growth and runaway internal thread churn. It is NOT an exception, so a
    plain try/except around the call never fires; the case just never returns and burns
    the whole T4 budget (reproduced: killed at 614s, 0 output).

    watershed_line=False on the identical input is proven correct and fast, so it is the
    safety-net fallback. A subprocess (not a thread or signal-based timeout) is required
    because the hang's internal thread growth cannot be reclaimed from within the same
    process -- only killing the process reliably reclaims it.
    """
    if not watershed_line:
        return watershed(elev, markers=markers, mask=mask, watershed_line=False)
    if timeout is None:
        timeout = float(os.environ.get('PENGWIN_WATERSHED_TIMEOUT', '8'))
    try:
        ctx = mp.get_context('fork')
        q = ctx.Queue()
        proc = ctx.Process(target=_watershed_worker, args=(q, elev, markers, mask))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            return watershed(elev, markers=markers, mask=mask, watershed_line=False)
        if not q.empty():
            return q.get()
        # child exited without a result (crashed/killed) -> safe fallback, never re-hang
        return watershed(elev, markers=markers, mask=mask, watershed_line=False)
    except Exception:
        # subprocess machinery itself misbehaved (e.g. restricted fork in some runtime)
        # -> fall back directly rather than risk the unguarded hang.
        return watershed(elev, markers=markers, mask=mask, watershed_line=False)


def _split_mask_by_clicks(
    mask: np.ndarray, points: list[tuple[int, int, int]], margin: int = 2,
    ct: np.ndarray = None, border: np.ndarray = None,  # [②] border = 모델 예측 골절면 확률(전체 볼륨)
    anatomy: str = None
) -> np.ndarray:
    """Watershed-split a single v19 component using one marker per click.

    The EDT + watershed run on the component's bounding box (padded by `margin` so a
    background shell surrounds it), not the full volume. This is identical to the
    full-volume result -- the distance field inside the component and the markers are
    unchanged -- but avoids running two O(volume) ops on a ~70M-voxel array for every
    merged component (the real Task2 wall-time hog: a badly under-segmented case has
    many clicks per v19 component, each split costing tens of seconds full-volume).
    """
    coords = np.argwhere(mask)
    shp = mask.shape
    full = np.zeros(shp, dtype=np.int32)
    if coords.size == 0:
        return full
    mins = np.maximum(coords.min(0) - margin, 0)
    maxs = np.minimum(coords.max(0) + 1 + margin, np.array(shp))
    sl = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    sub = mask[sl]

    markers = np.zeros(sub.shape, dtype=np.int32)
    sub_inds = None
    for idx, point in enumerate(points, start=1):
        lp = tuple(int(point[i] - mins[i]) for i in range(3))
        if all(0 <= lp[i] < sub.shape[i] for i in range(3)) and bool(sub[lp]):
            markers[lp] = idx
        else:
            if sub_inds is None:
                _, sub_inds = distance_transform_edt(~sub, return_indices=True)
            cp = tuple(int(np.clip(lp[i], 0, sub.shape[i] - 1)) for i in range(3))
            markers[tuple(int(sub_inds[ax][cp]) for ax in range(3))] = idx

    if int(markers.max()) <= 1:
        full[sl] = sub.astype(np.int32)
        return full
    n_expected = int(markers.max())
    distance = distance_transform_edt(sub)
    _wsl = os.environ.get('PENGWIN_WATERSHED_LINE', '0') == '1'
    # ============================================================================
    # [수정포인트 ②  — 점수 올리는 핵심 한 방]  by: 분석 결과
    # 진단: merge=1+split=1인데 f1=1.0(011/013/014/018)인 이유 = 클릭은 조각 '개수'를
    #       복구하지만(recall 1.0), 아래 분할이 '순수 EDT 기하'(두 마커의 기하학적
    #       중간면)로 잘라서 '면'이 틀림. 진짜 골절면은 넓은 사선 접촉면이라 seam을
    #       가로지르는 voxel이 오배정됨 -> topology 0.60~0.78에 갇힘.
    # 핵심: border-core 모델은 이 seam을 이미 알고 있음(merge 69쌍, border_cov 중앙값
    #       0.673, 65%가 >0.5). 즉 '신호'만 바꾸면 됨.
    # 할 일:
    #   1) v19_decode(inference_v7_d015.py)가 만드는 border-prob 채널(Dataset016
    #      7-class의 _border 클래스 2/4/6)을 relabel_from_clicks -> 여기로 plumb.
    #      (CT 배관 ct_arr_for_seam -> ct 와 '똑같은 경로'를 재사용하면 됨)
    #   2) 아래 elev 를  -distance + lambda*border_prob  로 교체(watershed_line=_wsl 유지).
    #   3) 28개 merge 케이스(proxy_out/merge_sites/part*.json)에서 lambda sweep + 게이트.
    # 주의: 이건 죽은 CT-seam과 '근본적으로 다름' — CT는 해면골 어두운 내부를 seam으로
    #       오인해 실패했지만, 학습된 border 채널엔 그 실패모드가 없음.
    # ============================================================================
    # [수정포인트 ①  — 죽은 길, 켜지 말 것] CT-seam: w=1.0/0.3 모두 baseline 미만(검증된 음수).
    #   pelvic 해면골 내부가 어두워 seam으로 오인. 코드/게이트는 정상이니 '구조만' 재사용하고
    #   PENGWIN_CT_SEAM 은 production 에서 절대 켜지 말 것. ②에서 신호를 border 로 갈아끼움.
    # [②] border-guided split: 모델이 예측한 골절면(border prob)을 절단면으로 사용.
    _border_seam = border is not None and os.environ.get('PENGWIN_BORDER_SEAM', '0') == '1'
    _ct_seam = ct is not None and os.environ.get('PENGWIN_CT_SEAM', '0') == '1'
    if _border_seam:
        # 능선(=watershed 경계)을 border 확률이 높은 곳(예측 골절면)에 강제.
        # markers 가 조각수를 고정 -> variant_a 의 과분할/phantom 모드 불가 -> 기하 앵커 불필요.
        bp = np.asarray(border[sl], dtype=np.float32)
        # [v65] ANCHORED elevation for hips. Measured (14-case sandbox + 15-case dumps):
        #   unanchored elev = bp        -> hips +0.0101 mean IoU but 6 cases WORSE, worst -0.562
        #   anchored  elev = -edtn+3*bp -> hips +0.0191 mean IoU, only 2 worse, worst -0.085
        # When bp ~ 0 the anchored form degenerates to today's exact -edtn behaviour, which is
        # why it restores every catastrophic regression (179 gt101 0.3716->0.9332) while keeping
        # the gains. Sacrum keeps the validated pure form (case001 frag2 IoU 0.013 -> 0.733).
        # CAUTION: predict_task2.py:193 does a hard `os.environ["PENGWIN_BORDER_PURE"]="1"`
        # (assignment, not setdefault), so reading that global here would silently force the
        # measured-BAD unanchored form onto hips. This repo has been bitten by exactly this
        # class of silently-inert env var three times already (v24..v30 COPY path,
        # CLICK_RECENTER, BORDER_SEAM), so the per-anatomy choice uses its own scoped keys
        # and never consults the global.
        # [v72] femur 를 hip 키에서 떼어낸다. v65 가 femur 를 hip 분기에 얹어놓아서,
        # femur 에 pure 를 켜려면 PENGWIN_BORDER_PURE_HIP=1 을 써야 하고 그러면 hip 도
        # 같이 pure 가 된다 — hip 의 pure 는 6케이스 악화 / worst -0.562 로 측정된 바로
        # 그 실패 모드(bp~0 컴포넌트가 평탄면이 되어 flood 가 degenerate)다. femur 는
        # bp 가 seam 중앙값 0.99 / interior 0.0 으로 갈려 그 조건이 성립하지 않는다.
        # 측정(v68, held-out 34케이스 x 3전략): base 0.7900 / anchored3.0 0.8593 / pure 0.8904.
        # 오늘 오라클(D013 femur 170, scripts/oracle_ceiling_oof.py)도 같은 방향이다:
        # 조각당 seed 한 점 + 순수 border 확률 elevation = f1 0.9832.
        # 이 변경은 harness_task2/click_guided_postproc_v73.py 에 이미 있었으나 배포된
        # 적이 없다. v73 은 이 4줄 외에 PENGWIN_BC_TAU 블록(135줄, 기본 off)을 같이
        # 들고 있었고 기각된 것은 그쪽이다 — 여기서는 femur 스코프 키만 떼어 옮긴다.
        if anatomy == 'femur':
            _pure = os.environ.get('PENGWIN_BORDER_PURE_FEMUR', '1') == '1'
        elif anatomy in ('leftHip', 'rightHip'):
            _pure = os.environ.get('PENGWIN_BORDER_PURE_HIP', '0') == '1'
        else:
            _pure = os.environ.get('PENGWIN_BORDER_PURE_SACRUM', '1') == '1'
        if _pure:
            # 검증된 recipe: variant_a_decode_anatomy(L242)/proxy_va_gate(L60)와 동일하게
            # border_prob 자체를 elevation 으로 사용 (거리항 없음, lambda 없음).
            # (mask=sub == v19 merged component, variant_a 의 mask=group 과 정확한 analogue)
            elev = bp
        else:
            # 비교용 하이브리드(미검증 lambda; -edtn 이 기하 midplane bias 재주입). pure 가 기본.
            lam  = float(os.environ.get('PENGWIN_BORDER_LAM', '3.0'))
            edtn = distance / (distance.max() + 1e-6)
            elev = -edtn + lam * bp
    elif _ct_seam:
        w_ct   = float(os.environ.get('PENGWIN_CT_W', '1.0'))
        w_dist = float(os.environ.get('PENGWIN_CT_WDIST', '0.5'))
        smooth = float(os.environ.get('PENGWIN_CT_SMOOTH', '1.0'))
        sub_ct = ct[sl]
        cs = gaussian_filter(sub_ct.astype(np.float32), smooth)
        fv = cs[sub]
        lo, hi = np.percentile(fv, 10), np.percentile(fv, 90)
        ctn = np.clip((cs - lo) / (hi - lo + 1e-6), 0, 1)
        edtn = distance / (distance.max() + 1e-6)
        # TODO[②]: 이 줄을 border 신호로 교체.  현재(죽은): CT 밝기.  목표:
        #   elev = -edtn + lam * border_prob[sl][...]   # border = 모델 예측 골절면
        elev = w_ct * (1.0 - ctn) - w_dist * edtn  # <- [죽은 신호] CT 밝기, 재시도 금지
    else:
        # 기본 경로(현재 production). 순수 EDT 기하 분할 = '면'이 틀려서 merge topology가
        # 깎임. ②가 적용되면 여기 markers 분할도 border 신호를 받게 바뀌어야 함.
        elev = -distance

    # [수정포인트 ④ — 조건부(GT-free) seed-anchor: v32 대비 v33은 case001류(클릭이
    # 경계 근처라 merge)는 고쳤지만 157/179/184/186처럼 원래 문제없던 케이스에서
    # 앵커가 불필요하게 경계를 밀어 정밀도가 떨어졌다(14케이스 실측: F1 0.931->0.914).
    # 두 버전을 전부 쓰되, "merge가 실제로 감지된 컴포넌트에만" anchor를 적용해서
    # 좋은 점만 취한다 -- v53 task1 컨센서스와 동일한 GT-free per-component 선택.
    # [v72] 시도했다가 걷어낸 것 — sliver seed 게이트. anchored 로 먼저 나눠 1cm3 에
    # 못 미치는 마커를 지운 뒤 pure 를 돌리면, pure 가 작은 마커를 부풀려 채점되는
    # 거짓양성으로 만드는 케이스(277: 예측 7 -> 4 로 f1 0.727 -> 1.000, 278: 4 -> 2 로
    # 0.667 -> 1.000)는 확실히 고쳐진다. 그런데 **전수에서는 해롭다**:
    #   170케이스 x 4전략(680쌍)  anchored 0.9081 / pure 0.9167 / 게이트 0.8850
    #   recall 0.8906 / 0.9188 / 0.8477
    # 전제가 틀렸다. "anchored 에서 작은 마커 = 채점 안 될 슬리버"가 아니다 — anchored 는
    # 진짜 조각도 굶긴다(이 파일이 이미 아는 anchor-starvation, PENGWIN_STARVE_FLOOR).
    # 게이트는 그렇게 굶은 진짜 조각을 통째로 지워 recall 을 0.07 깎았다. 재시도 금지.
    result = _watershed_safe(elev, markers, sub, _wsl).astype(np.int32)
    # [v56 fix] v34's retry trigger was `n_got < n_expected` -- "did every
    # click produce a DISTINCT label" -- which can practically never fire,
    # since marker-seeded watershed always keeps at least the 1-voxel seed
    # itself, so a label counts as "present" even when starved to nothing.
    # Confirmed via case001 internal_boundary: frag2 (GT 6366vox) shrank to
    # 1 voxel, n_got was still 3==n_expected, so the anchor never triggered
    # even though PENGWIN_SEED_ANCHOR_R was set. Fixed to a size-based check
    # (any label below a floor), matching the fix later versions independently
    # converged on -- the floor value (20) and its calibration rationale
    # (between the smallest observed real fragment ~26vox and observed
    # starvation failures ~1-4vox) are unchanged from that later work.
    # [v61] floor is now tunable via PENGWIN_STARVE_FLOOR (int, sweep) or
    # 'auto' (component-relative: scales with how much territory each click
    # would get under a perfectly even split, instead of one fixed global
    # constant). fair_share = component voxels / n_expected clicks; auto floor
    # = clip(fair_share * PENGWIN_STARVE_FRAC, 4, 26) using the same empirical
    # bounds (4 = worst observed starvation failure, 26 = smallest observed
    # real fragment) that anchored the original fixed value of 20.
    _floor_setting = os.environ.get('PENGWIN_STARVE_FLOOR', '20')
    if _floor_setting == 'auto':
        _frac = float(os.environ.get('PENGWIN_STARVE_FRAC', '0.15'))
        _fair_share = int(sub.sum()) / max(n_expected, 1)
        _STARVE_FLOOR = int(np.clip(_fair_share * _frac, 4, 26))
    else:
        _STARVE_FLOOR = int(_floor_setting)
    sizes_by_label = {idx: int((result == idx).sum()) for idx in range(1, n_expected + 1)}
    n_got = len(np.unique(result[result > 0]))
    _anchor_r = int(os.environ.get('PENGWIN_SEED_ANCHOR_R', '0'))
    _any_starved = any(sz < _STARVE_FLOOR for sz in sizes_by_label.values())
    if _anchor_r > 0 and (n_got < n_expected or _any_starved):
        # merge 감지(voxel 사실만으로 판정, GT 불필요): 클릭 개수만큼 label이 안 나옴.
        # anchor로 마커를 부풀려 재시도 -- elev는 그대로 재사용, markers만 교체.
        anchored = markers.copy()
        for idx in range(1, n_expected + 1):
            seed = anchored == idx
            if not seed.any():
                continue
            grown = binary_dilation(seed, structure=_CONN26, iterations=_anchor_r)
            grown &= sub & (anchored == 0)
            anchored[grown] = idx
        retried = _watershed_safe(elev, anchored, sub, _wsl).astype(np.int32)
        # [v56 fix, part 2] the acceptance check had the SAME flaw as the
        # trigger: comparing distinct-label COUNTS, which can't detect "the
        # starved label grew from 1 voxel to 6923" since the count of
        # present labels doesn't change (both before and after are 3/3).
        # Accept the retry if it didn't lose any previously-present label
        # and grew at least one previously-starved label.
        retried_sizes = {idx: int((retried == idx).sum()) for idx in range(1, n_expected + 1)}
        no_label_lost = all(retried_sizes[idx] > 0 for idx, sz in sizes_by_label.items() if sz > 0)
        starved_grew = any(retried_sizes[idx] > sizes_by_label[idx]
                            for idx, sz in sizes_by_label.items() if sz < _STARVE_FLOOR)
        if len(np.unique(retried[retried > 0])) > n_got or (no_label_lost and starved_grew):
            result = retried

    full[sl] = result
    return full


def _absorb_residual(arr: np.ndarray, out: np.ndarray, anat_mask: np.ndarray, lo: int, hi: int) -> None:
    """Absorb unclicked v19 shards that touch a clicked fragment; drop isolated ones.

    Recovers over-segmentation (a clicked fragment that v19 split into pieces) while
    removing isolated false positives (anatomy voxels no click selected).
    """
    assigned = (out >= lo) & (out <= hi)
    residual = anat_mask & ~assigned
    if not residual.any() or not assigned.any():
        return
    components, n = cc_label(residual, structure=_CONN26)
    for i in range(1, n + 1):
        cc = components == i
        rim = binary_dilation(cc, structure=_CONN26) & assigned
        if not rim.any():
            continue  # isolated -> drop
        rim_ids = out[rim]
        rim_ids = rim_ids[(rim_ids >= lo) & (rim_ids <= hi)]
        if rim_ids.size:
            ids, counts = np.unique(rim_ids, return_counts=True)
            out[cc] = ids[int(counts.argmax())]


def relabel_from_clicks(instance_arr: np.ndarray, clicks: list[Click], image=None, ct_arr=None, border_arr=None) -> np.ndarray:
    """Map a v19 instance prediction to click-selected PENGWIN fragments (1-200).

    `image` (the SimpleITK image whose array is `instance_arr`) is optional; when given,
    click points are also interpreted as physical/world coordinates, so the resolver is
    robust whether Grand Challenge sends voxel indices or world-mm coordinates.
    """
    arr = instance_arr
    ct_arr_for_seam = ct_arr
    border_for_seam = border_arr  # [②] border prob 볼륨 -> _split_mask_by_clicks 로 전달
    shape = arr.shape  # type: ignore[assignment]
    out = np.zeros(shape, dtype=np.uint8)

    masks: dict[str, np.ndarray] = {}
    edt_cache: dict[str, Optional[tuple[np.ndarray, np.ndarray]]] = {}

    def get_mask(anatomy: str) -> np.ndarray:
        if anatomy not in masks:
            lo, hi = ANATOMY_RANGES[anatomy]
            masks[anatomy] = (arr >= lo) & (arr <= hi)
        return masks[anatomy]

    def edt_getter(key: str, mask: np.ndarray):
        def _get():
            if key not in edt_cache:
                edt_cache[key] = (
                    distance_transform_edt(~mask, return_indices=True) if mask.any() else None
                )
            return edt_cache[key]
        return _get

    full_fg: Optional[np.ndarray] = None

    # Pass 1: resolve every click to (anatomy, voxel, v19 source label). The EDT is
    # built lazily per anatomy and only touched when a click lands outside the bone
    # under BOTH axis interpretations (rare) -- keeps large cases fast.
    resolved: dict[str, list[tuple[tuple[int, int, int], int]]] = {a: [] for a in ANATOMY_RANGES}
    for click in clicks:
        anatomy = click.anatomy
        if anatomy in ANATOMY_RANGES:
            mask = get_mask(anatomy)
            if not mask.any():
                continue  # v19 found nothing for this anatomy
            vox = _resolve_voxel_in_mask(click.point, shape, mask, edt_getter(anatomy, mask), image=image)
            if vox is None:
                continue
            src = int(arr[vox])
            lo, hi = ANATOMY_RANGES[anatomy]
            if lo <= src <= hi:
                resolved[anatomy].append((vox, src))
        else:
            # No anatomy in the name: resolve against the whole foreground and read it off.
            if full_fg is None:
                full_fg = arr > 0
            if not full_fg.any():
                continue
            vox = _resolve_voxel_in_mask(click.point, shape, full_fg, edt_getter("__fg__", full_fg), image=image)
            if vox is None:
                continue
            src = int(arr[vox])
            inferred = _range_to_anatomy(src)
            if inferred is not None:
                resolved[inferred].append((vox, src))

    # Pass 2: per anatomy, group clicks by v19 source label and emit fragments.
    for anatomy, (lo, hi) in ANATOMY_RANGES.items():
        items = resolved[anatomy]
        if not items:
            continue
        mask = get_mask(anatomy)
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for vox, src in items:
            groups.setdefault(src, []).append(vox)

        next_id = lo
        for src, voxes in groups.items():
            src_mask = arr == src
            if not src_mask.any():
                continue
            if len(voxes) == 1:
                if next_id <= hi:
                    out[src_mask] = next_id
                    next_id += 1
                continue
            # [②] border_arr 가 anatomy별 dict 이면 현재 anatomy 채널만 선택(좌우 hip 누수 방지).
            #     단일 배열/ None 은 그대로 통과(femur -> None -> EDT 기본 경로).
            # [v57 fix] PENGWIN_BORDER_SEAM=1 broke leftHip/rightHip catastrophically
            # (IoU 0.97->0.096, most tissue vanished / sides swapped) on RAS-oriented
            # case002 -- confirmed via isolation test that this happens with the
            # border-seam env var ALONE, unrelated to the anchor-trigger fix. Root
            # cause of the hip-specific reorientation issue not yet found; scope
            # border-seam to sacrum only (where it was diagnosed and validated) until
            # hip is separately investigated and re-enabled.
            # [v65] The v57 sacrum-only gate was installed to prevent a hip catastrophe that
            # MEASURABLY NEVER HAPPENED: hip anatomy IoU is 0.9861/0.9864 in all 80 recorded
            # case-002 runs v34..v58, and the seam-ALL vs seam-sacrum isolation test differs by
            # 16 / 36 voxels out of 38,413,956. The quoted "0.97 -> 0.096" matches the SACRUM
            # anchor-starvation failure, and "sides swapped" traces to a v45 rightHip 102/103
            # ID permutation. Meanwhile the gate costs real accuracy: sacrum-only seam is
            # bit-identical to seam-OFF on all 10 metrics, i.e. it does nothing at all, while
            # hips carry a BETTER seam signal than sacrum (AUC 0.9983 vs 0.9954).
            _seam_anats = set(
                os.environ.get('PENGWIN_SEAM_ANATS', 'sacrum,leftHip,rightHip').split(','))
            if isinstance(border_for_seam, dict) and anatomy in _seam_anats:
                border_sel = border_for_seam.get(anatomy)
            else:
                border_sel = None
            split = _split_mask_by_clicks(src_mask, voxes, ct=ct_arr_for_seam, border=border_sel,
                                          anatomy=anatomy)
            for split_id in range(1, int(split.max()) + 1):
                if next_id > hi:
                    break
                    
                part = split == split_id
                if part.any():
                    out[part] = next_id
                    next_id += 1

        _absorb_residual(arr, out, mask, lo, hi)

    return out
