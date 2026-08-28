#!/usr/bin/env python3
"""
PENGWIN Task1 — 340 proxy 평가 harness  (progress_report_103 §3 구현)

목적
----
public-5(가려진 데이터)를 직접 튜닝하지 않고, public-5가 보여준 *failure type*을
340 held-out CV에서 재현해 "안 보이는 문제 → 보이는 문제"로 바꾼다.

설계 원칙 (중요)
----------------
1. 지표는 재구현하지 않는다. 공식 평가기(pengwin_eval, hungarian/panoptic + LDSC)를
   유일한 진실원으로 *wrapping*한다.  → report 100이 지적한 평가기 2개 0.057 불일치를
   "공식 것으로 일원화"하여 닫는다. oof_11metric.py(greedy IoU≥0.5)는 폐기한다.
2. harness는 그 위에 메타데이터 / 층화(parts-bin, bone) / failure-type 태깅 /
   집계 / sentinel 추적 / run 비교만 얹는다.
3. 일원화 증명: 이미 제출해 받은 public-5 결과 JSON을 harness가 그대로 재현하는지
   validate_against_reference()로 확인한다. 재현되면 "harness eval == 공식 eval"이 증명된다.

이 harness는 held-out CV 예측만 채점한다. public-5 raw 데이터는 건드리지 않는다
(public-5는 오직 validate_against_reference의 reference JSON으로만 들어온다).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import SimpleITK as sitk

# repo 루트를 sys.path에 보장 — 어디서 실행하든(예: 다른 cwd/스크립트 경로)
# `evaluation.*` / `scripts.*` import가 깨지지 않도록 한다.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 0. 라벨 인코딩 (challenge 스펙)
# ─────────────────────────────────────────────────────────────────────────────
BONE_BLOCKS = {
    "sacrum":    (1, 50),
    "left_hip":  (51, 100),
    "right_hip": (101, 150),
    "femur":     (151, 200),   # report 100: femur는 HD95가 터지는 곳 → 반드시 포함
}
OFFICIAL_MIN_FRAGMENT_VOLUME_MM3 = 500.0  # challenge: <500mm^3 fragment는 채점서 omit

# headline 지표 키 — public-5 / 리더보드 JSON과 *완전히 동일한 이름*이어야 비교 가능
METRIC_KEYS = [
    "ins_recall", "ins_precision", "ins_f1_score",
    "merge_error_count", "split_error_count",
    "fracture_dice", "fracture_local_dice", "fracture_hd95", "fracture_assd",
    "topology_consistency",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HarnessConfig:
    data_root: Path                 # raw_data/PENGWIN_train (GT label.mha 위치)
    pred_root: Path                 # held-out CV 예측 (.mha) 위치
    out_dir: Path = Path("./proxy_out")
    cases: Optional[list[str]] = None   # held-out case id 목록. None이면 pred_root 전체
    min_fragment_volume_mm3: float = OFFICIAL_MIN_FRAGMENT_VOLUME_MM3

    # failure-type 태깅 임계 (calibratable). 기본값은 public-5 매핑을 재현하도록 잡음.
    ldsc_good: float = 0.93         # 이 이상이면 surface 양호
    hd95_good: float = 3.0          # mm. 이 이하면 surface 양호
    shared_recall_max: float = 0.75 # 이 이하 + 고HD95 + merge>0 → 001형 shared-limit 후보
    shared_hd95_min: float = 12.0   # mm

    # 파일명 패턴 (repo에 맞게 조정)
    gt_pattern: str = "{case}/label.mha"
    pred_pattern: str = "{case}.mha"
    eval_mode: str = "strict-official"  # strict-official | local-composite


# ─────────────────────────────────────────────────────────────────────────────
# 2. 공식 평가기 어댑터 — 유일한 진실원. 반드시 repo에 wiring할 것.
# ─────────────────────────────────────────────────────────────────────────────
#
#  아래 둘 중 하나로 연결한다. 절대 직접 재구현하지 말 것(세 번째 불일치를 만든다).
#
#  (A) 권장 — 공식 평가 모듈의 per-case 함수를 import:
#      from evaluation import evaluate_case as _OFFICIAL_EVAL   # PENGWIN2026 baseline repo
#      def official_score_case(gt_path, pred_path): return _OFFICIAL_EVAL(pred_path, gt_path)
#
#  (B) 공식 코드가 CLI 스크립트뿐이면 subprocess로 호출 후 JSON 파싱.
#
#  주의: 여기서 쓰는 평가기는 grand-challenge가 public-5 JSON을 만든 그 코드와
#  *동일*해야 한다(특히 fracture_local_dice = local_dice_cfs20, hungarian/panoptic 매칭,
#  topology_consistency). 로컬 pengwin_eval.py가 구버전이면 PENGWIN2026 공식 eval로 동기화.

_OFFICIAL_EVAL: Optional[Callable[[str, str], dict]] = None  # ← repo에서 주입: (pred_path, gt_path)


def official_score_case(gt_path: Path, pred_path: Path) -> dict:
    """공식 per-case 평가기를 호출해 METRIC_KEYS + num_parts를 담은 dict를 돌려준다."""
    if CURRENT_EVAL_MODE == "local-composite":
        return local_composite_score_case(gt_path, pred_path)
    if _OFFICIAL_EVAL is None:
        raise NotImplementedError(
            "공식 평가기를 wiring하세요 (위 (A)/(B) 참조). "
            "harness는 지표를 재구현하지 않습니다 — 공식 eval이 유일 진실원입니다."
        )
    res = _OFFICIAL_EVAL(str(pred_path), str(gt_path))
    # 공식 반환에 빠진 키가 있으면 즉시 시끄럽게 실패(조용한 잘못된 비교 방지)
    missing = [k for k in METRIC_KEYS if k not in res]
    if missing:
        raise KeyError(f"공식 eval 반환에 키 누락: {missing}. 공식 eval 버전을 확인하세요.")
    return res


CURRENT_EVAL_MODE = "strict-official"


def normalize_local_composite_metrics(instance_metrics: dict, official_like_metrics: dict) -> dict:
    """Map repo-local evaluator outputs into the proxy harness schema.

    This is intentionally marked local-composite rather than official: instance/topology
    comes from scripts.oof_11metric, while LDSC/HD95/ASSD comes from evaluation.pengwin_eval.
    Use it for 340 proxy exploration, not for proving public-5 official equivalence.
    """
    return {
        "ins_recall": float(instance_metrics["instance_recall"]),
        "ins_precision": float(instance_metrics["instance_precision"]),
        "ins_f1_score": float(instance_metrics["instance_f1"]),
        "merge_error_count": float(instance_metrics["merge_errors"]),
        "split_error_count": float(instance_metrics["split_errors"]),
        "fracture_dice": float(instance_metrics["score_dice"]),
        "fracture_local_dice": float(official_like_metrics["LDSC_F"]),
        "fracture_hd95": float(official_like_metrics["HD95_F_mm"]),
        "fracture_assd": float(official_like_metrics["ASSD_F_mm"]),
        "topology_consistency": float(instance_metrics["topology_consistency"]),
    }


def local_composite_score_case(gt_path: Path, pred_path: Path) -> dict:
    """Local proxy adapter for development when official 2026 per-case JSON is unavailable."""
    from evaluation.pengwin_eval import evaluate_case as eval_surface_case
    from scripts.oof_11metric import score_case as eval_instance_case

    instance_metrics = eval_instance_case(str(pred_path), str(gt_path), do_hd95=False)
    if "error" in instance_metrics:
        raise RuntimeError(instance_metrics["error"])
    # evaluation.pengwin_eval.load_label_volume does not reorient images; align
    # both volumes to LPS here so surface metrics match the instance evaluator.
    with tempfile.TemporaryDirectory(prefix="pengwin_surface_lps_") as td:
        tmp = Path(td)
        pred_lps = sitk.DICOMOrient(sitk.ReadImage(str(pred_path)), "LPS")
        gt_lps = sitk.DICOMOrient(sitk.ReadImage(str(gt_path)), "LPS")
        pred_lps_path = tmp / "pred.mha"
        gt_lps_path = tmp / "gt.mha"
        sitk.WriteImage(pred_lps, str(pred_lps_path), useCompression=True)
        sitk.WriteImage(gt_lps, str(gt_lps_path), useCompression=True)
        surface_metrics = eval_surface_case(
            str(pred_lps_path),
            str(gt_lps_path),
            matching_method="hungarian",
            min_cc_mm3=1000.0,
        )
    mapped = normalize_local_composite_metrics(instance_metrics, surface_metrics)
    missing = [k for k in METRIC_KEYS if k not in mapped]
    if missing:
        raise KeyError(f"local-composite adapter missing keys: {missing}")
    return mapped


def official_score_bonewise(gt_path: Path, pred_path: Path) -> dict[str, dict]:
    """bone별 지표 = 해당 bone label block만 남기고 공식 eval 재호출(진실원 재사용).

    다른 block을 0으로 마스킹한 임시 .mha를 만들어 official_score_case에 통과시킨다.
    """
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)
    out = {}
    with tempfile.TemporaryDirectory(prefix="pengwin_bonewise_") as td:
        tmp = Path(td)
        for bone, (lo, hi) in BONE_BLOCKS.items():
            if not ((gt >= lo) & (gt <= hi)).any():
                continue  # 이 케이스에 해당 bone 없음 (pelvic/femur 배타)
            g = np.where((gt >= lo) & (gt <= hi), gt, 0)
            p = np.where((pred >= lo) & (pred <= hi), pred, 0)
            g_path = tmp / f"gt_{bone}.mha"
            p_path = tmp / f"pred_{bone}.mha"
            _write_like(g, gt_img, g_path)
            _write_like(p, pred_img, p_path)
            try:
                out[bone] = official_score_case(g_path, p_path)
            except Exception as e:  # bone에 fragment 0개 등 — 진단치이므로 관대하게
                out[bone] = {"error": str(e)}
    return out


def _write_like(arr: np.ndarray, ref: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.uint8 if arr.max() < 256 else np.int16))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path), useCompression=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GT 메타데이터 — num_parts(부피 omit 적용), parts-bin, bone별 fragment 수, spacing
# ─────────────────────────────────────────────────────────────────────────────
def case_metadata(gt_path: Path, min_vol: float) -> dict:
    img = sitk.ReadImage(str(gt_path))
    arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()
    voxel_vol = sx * sy * sz

    bone_counts: dict[str, int] = {}
    total = 0
    case_type = None
    for bone, (lo, hi) in BONE_BLOCKS.items():
        labels = [l for l in np.unique(arr) if lo <= l <= hi]
        kept = 0
        for l in labels:
            vol = (arr == l).sum() * voxel_vol
            if vol >= min_vol:        # 공식 omit 규칙과 일치
                kept += 1
        bone_counts[bone] = kept
        total += kept
        if kept > 0:
            case_type = "femur" if bone == "femur" else "pelvic"
    # pelvic/femur 배타(report 101): femur 외 block이 있으면 pelvic
    if any(bone_counts[b] > 0 for b in ("sacrum", "left_hip", "right_hip")):
        case_type = "pelvic"

    return {
        "num_parts": total,
        "parts_bin": parts_bin(total),
        "bone_fragment_counts": bone_counts,
        "case_type": case_type,
        "spacing_xyz": (round(sx, 4), round(sy, 4), round(sz, 4)),
        "voxel_volume_mm3": round(voxel_vol, 5),
    }


def parts_bin(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


# ─────────────────────────────────────────────────────────────────────────────
# 4. failure-type 태깅 — public-5 매핑을 재현하도록 규칙화 (투명·calibratable)
# ─────────────────────────────────────────────────────────────────────────────
def tag_failure_type(m: dict, cfg: HarnessConfig) -> str:
    """
    public-5 기대 매핑:
      004(parts1, instance완벽, LDSC0.86/HD8.67) -> surface
      003(recall0.92, merge0/split0)            -> merge_sep  (조각 누락/병합)
      001(recall0.73, merge1.0, HD17)           -> shared_limit (후보)
      002/005(instance완벽, LDSC↑ HD↓)          -> solved
    """
    recall = m["ins_recall"]
    merge = m["merge_error_count"]
    split = m["split_error_count"]
    ldsc = m["fracture_local_dice"]
    hd95 = m["fracture_hd95"]

    instance_clean = (recall >= 1.0 - 1e-9) and (merge <= 1e-9) and (split <= 1e-9)
    surface_ok = (ldsc >= cfg.ldsc_good) and (hd95 <= cfg.hd95_good)

    # 001형: 공동 한계 후보 (심한 merge + 매우 낮은 recall + 매우 큰 HD95)
    if (merge > 1e-9) and (recall <= cfg.shared_recall_max) and (hd95 >= cfg.shared_hd95_min):
        return "shared_limit"
    # 과분할 우세
    if split > merge and split > 1e-9:
        return "split_overseg"
    # 003형: 분리 실패 (병합 또는 조각 누락)
    if (merge > 1e-9) or (recall < 1.0 - 1e-9):
        return "merge_sep"
    # 여기부터 instance는 깨끗함
    if instance_clean and not surface_ok:
        return "surface"          # 004형
    if instance_clean and surface_ok:
        return "solved"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# 5. per-case 채점 = 공식 지표 + 메타 + 태그 + (옵션) bone별
# ─────────────────────────────────────────────────────────────────────────────
def score_case(case: str, cfg: HarnessConfig, bonewise: bool = True) -> dict:
    global CURRENT_EVAL_MODE
    previous_eval_mode = CURRENT_EVAL_MODE
    CURRENT_EVAL_MODE = cfg.eval_mode
    gt_path = cfg.data_root / cfg.gt_pattern.format(case=case)
    pred_path = cfg.pred_root / cfg.pred_pattern.format(case=case)
    try:
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)

        metrics = official_score_case(gt_path, pred_path)
        meta = case_metadata(gt_path, cfg.min_fragment_volume_mm3)
        rec = {
            "case_id": case,
            **{k: metrics[k] for k in METRIC_KEYS},
            "num_parts": meta["num_parts"],
            "parts_bin": meta["parts_bin"],
            "case_type": meta["case_type"],
            "bone_fragment_counts": meta["bone_fragment_counts"],
            "spacing_xyz": meta["spacing_xyz"],
        }
        rec["failure_type"] = tag_failure_type(rec, cfg)
        if bonewise:
            bw = official_score_bonewise(gt_path, pred_path)
            rec["bonewise"] = {
                b: {k: v.get(k) for k in METRIC_KEYS} if "error" not in v else v
                for b, v in bw.items()
            }
        return rec
    finally:
        CURRENT_EVAL_MODE = previous_eval_mode


# ─────────────────────────────────────────────────────────────────────────────
# 6. 집계 — overall / parts-bin / bone / failure-type
# ─────────────────────────────────────────────────────────────────────────────
def _mean(vals: list[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.mean(vals)), 6) if vals else None


def _agg(records: list[dict]) -> dict:
    return {"num": len(records), **{k: _mean([r[k] for r in records]) for k in METRIC_KEYS}}


def aggregate(records: list[dict]) -> dict:
    out = {"overall": _agg(records)}

    by_bin = defaultdict(list)
    for r in records:
        by_bin[r["parts_bin"]].append(r)
    out["by_parts_bin"] = {b: _agg(rs) for b, rs in sorted(by_bin.items())}

    by_tag = defaultdict(list)
    for r in records:
        by_tag[r["failure_type"]].append(r)
    out["by_failure_type"] = {t: _agg(rs) for t, rs in sorted(by_tag.items())}
    out["failure_type_cases"] = {t: [r["case_id"] for r in rs] for t, rs in sorted(by_tag.items())}

    # bone별: 각 case의 bonewise를 모아 평균
    by_bone = defaultdict(list)
    for r in records:
        for bone, bm in r.get("bonewise", {}).items():
            if "error" not in bm:
                by_bone[bone].append(bm)
    out["by_bone"] = {b: {"num": len(ms), **{k: _mean([m[k] for m in ms]) for k in METRIC_KEYS}}
                      for b, ms in sorted(by_bone.items())}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. sentinel 추적 + run 비교 (학습/decode A/B의 판정판)
# ─────────────────────────────────────────────────────────────────────────────
def sentinel_view(records: list[dict], sentinels: dict[str, list[str]]) -> dict:
    by_id = {r["case_id"]: r for r in records}
    out = {}
    for group, ids in sentinels.items():
        out[group] = [
            {kk: by_id[c].get(kk) for kk in ["case_id", "failure_type", "ins_recall",
             "merge_error_count", "split_error_count", "fracture_local_dice",
             "fracture_hd95", "topology_consistency"]}
            for c in ids if c in by_id
        ]
    return out


def compare_runs(base: dict, cand: dict, cfg: HarnessConfig) -> dict:
    """두 run의 집계를 받아 delta + failure-type별 판정.

    학습 레버(예: critical seam-gap loss)의 성공 기준:
      merge↓  AND  split 안 터짐  AND  topology_consistency↑  AND  LDSC/HD95 안 망가짐
    surface 레버의 성공 기준:
      surface bin에서 LDSC↑ / HD95↓  (merge_sep 망가뜨리지 않고)
    """
    def d(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 6)

    verdict = {}
    bm, cm = base["overall"], cand["overall"]
    overall_delta = {k: d(bm[k], cm[k]) for k in METRIC_KEYS}

    def group_delta(group: str, metric: str) -> Optional[float]:
        return d(base["by_failure_type"].get(group, {}).get(metric),
                 cand["by_failure_type"].get(group, {}).get(metric))

    merge_sep_improved = (
        ((group_delta("merge_sep", "merge_error_count") or 0) < 0) or
        ((group_delta("merge_sep", "ins_recall") or 0) > 0)
    )
    merge_sep_guard = (
        (group_delta("merge_sep", "split_error_count") or 0) <= 1e-6 and
        (group_delta("merge_sep", "fracture_local_dice") or 0) >= -1e-3 and
        (group_delta("merge_sep", "fracture_hd95") or 0) <= 1e-3 and
        (group_delta("merge_sep", "topology_consistency") or 0) >= -1e-3
    )
    overall_guard = (
        (overall_delta["split_error_count"] or 0) <= 1e-6 and
        (overall_delta["fracture_local_dice"] or 0) >= -1e-3 and
        (overall_delta["fracture_hd95"] or 0) <= 1e-3
    )
    # merge/seam 축 판정 (003형): 실제 merge_sep 개선이 있어야 pass.
    merge_ok = merge_sep_improved and merge_sep_guard and overall_guard
    verdict["merge_axis_pass"] = bool(merge_ok)

    # surface 축 판정 (004형) — by_failure_type에 surface가 있으면
    bs = base["by_failure_type"].get("surface")
    cs = cand["by_failure_type"].get("surface")
    if bs and cs:
        verdict["surface_axis_pass"] = bool(
            (d(bs["fracture_local_dice"], cs["fracture_local_dice"]) or 0) >= 0 and
            (d(bs["fracture_hd95"], cs["fracture_hd95"]) or 0) <= 0
        )

    return {
        "overall_delta": overall_delta,
        "by_failure_type_delta": {
            t: {k: d(base["by_failure_type"].get(t, {}).get(k),
                     cand["by_failure_type"].get(t, {}).get(k)) for k in METRIC_KEYS}
            for t in set(base["by_failure_type"]) | set(cand["by_failure_type"])
        },
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. 일원화 증명 — public-5 제출 JSON을 harness가 재현하는지 검증
# ─────────────────────────────────────────────────────────────────────────────
def validate_against_reference(reference_json: Path, public5_gt_root: Path,
                               public5_pred_root: Path, cfg: HarnessConfig,
                               atol: float = 1e-3) -> dict:
    """이미 제출해 받은 public-5 결과 JSON과, 같은 예측을 harness로 채점한 값을 비교.
    재현되면 harness eval == 공식 eval 이 증명된다(평가기 일원화 완료).
    public5_*_root는 sanity-check 용도로 public-5 예측/GT가 로컬에 있을 때만.
    """
    ref = json.loads(Path(reference_json).read_text())
    ref_by_id = {r["case_id"]: r for r in ref["results"]}
    sub_cfg = HarnessConfig(
        data_root=public5_gt_root,
        pred_root=public5_pred_root,
        out_dir=cfg.out_dir,
        min_fragment_volume_mm3=cfg.min_fragment_volume_mm3,
        gt_pattern=cfg.gt_pattern,
        pred_pattern=cfg.pred_pattern,
        eval_mode=cfg.eval_mode,
    )
    mismatches = []
    for case, ref_m in ref_by_id.items():
        rec = score_case(case, sub_cfg, bonewise=False)
        for k in METRIC_KEYS:
            if k in ref_m and ref_m[k] is not None and rec[k] is not None:
                if abs(rec[k] - ref_m[k]) > atol:
                    mismatches.append({"case": case, "metric": k,
                                       "harness": rec[k], "reference": ref_m[k]})
    return {"unified": len(mismatches) == 0, "n_mismatch": len(mismatches),
            "mismatches": mismatches}


# ─────────────────────────────────────────────────────────────────────────────
# 9. 출력 — JSON + markdown 요약
# ─────────────────────────────────────────────────────────────────────────────
def to_markdown(agg: dict) -> str:
    def row(name, a):
        return (f"| {name} | {a['num']} | {a['ins_recall']} | {a['merge_error_count']} "
                f"| {a['split_error_count']} | {a['fracture_local_dice']} "
                f"| {a['fracture_hd95']} | {a['topology_consistency']} |")
    head = ("| group | n | recall | merge | split | LDSC | HD95 | topo |\n"
            "|---|--:|--:|--:|--:|--:|--:|--:|")
    lines = [head, row("overall", agg["overall"])]
    for b, a in agg["by_parts_bin"].items():
        lines.append(row(f"parts={b}", a))
    for t, a in agg["by_failure_type"].items():
        lines.append(row(f"type={t}", a))
    for b, a in agg["by_bone"].items():
        lines.append(row(f"bone={b}", a))
    return "\n".join(lines)


def _load_checkpoint(path: Path) -> list:
    """records.jsonl에서 *성공한* per-case 레코드만 복원(case_id별 최신 유지).
    에러 레코드는 일시적일 수 있으므로 done으로 치지 않고 재시도 대상으로 둔다.
    손상된(잘린) 마지막 줄은 무시.
    """
    by_id = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_kind") == "error":
            continue
        cid = obj.get("case_id")
        if cid is not None:
            by_id[cid] = obj
    return list(by_id.values())


def run(cfg: HarnessConfig, sentinels: Optional[dict] = None, bonewise: bool = True,
        resume: bool = True) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cases = cfg.cases or sorted({p.stem for p in cfg.pred_root.glob("*.mha")})

    ckpt = cfg.out_dir / "records.jsonl"
    records = _load_checkpoint(ckpt) if resume else []
    done = {r["case_id"] for r in records}
    todo = [c for c in cases if c not in done]
    if done:
        print(f"[resume] {len(done)} case(s) 성공 기록 있음, {len(todo)} case(s) 남음", flush=True)

    # 체크포인트를 성공 레코드만으로 정리(에러/중복 줄 제거) 후 이어 append
    with ckpt.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()

    errors = []
    with ckpt.open("a") as fh:
        for i, c in enumerate(todo, 1):
            try:
                rec = score_case(c, cfg, bonewise=bonewise)
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                err = {"case": c, "error": str(e)}
                errors.append(err)
                fh.write(json.dumps({"_kind": "error", **err}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {c} done", flush=True)

    agg = aggregate(records)
    report = {"config": {**asdict(cfg), "data_root": str(cfg.data_root),
                          "pred_root": str(cfg.pred_root), "out_dir": str(cfg.out_dir)},
              "n_cases": len(records), "errors": errors,
              "aggregates": agg, "results": records}
    if sentinels:
        report["sentinels"] = sentinel_view(records, sentinels)

    (cfg.out_dir / "proxy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (cfg.out_dir / "proxy_summary.md").write_text(to_markdown(agg))
    print(to_markdown(agg))
    if errors:
        print(f"\n[warn] {len(errors)} case(s) 실패: {[e['case'] for e in errors]}")
    return report


# 기본 sentinel — report 102/103. (이건 내부 case id; public-5 id와 헷갈리지 말 것)
DEFAULT_SENTINELS = {
    "hard_pelvic":     ["084", "179", "064", "198", "115", "157"],
    "worst_border":    ["025", "110", "036", "101"],
    "sacrum_heavy":    ["084", "064", "198", "075", "006"],
    "right_heavy":     ["180", "157", "186", "101"],
}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PENGWIN Task1 340 proxy harness")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("./proxy_out"))
    ap.add_argument("--cases", type=Path, default=None,
                    help="held-out case id 목록 파일(줄당 1개). 없으면 pred-root 전체")
    ap.add_argument("--no-bonewise", action="store_true")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="baseline proxy_report.json — 이 run과 delta/verdict 비교")
    ap.add_argument("--validate-ref", type=Path, default=None,
                    help="public-5 결과 JSON — harness 재현 검증(평가기 일원화 증명)")
    ap.add_argument("--public5-gt-root", type=Path, default=None)
    ap.add_argument("--public5-pred-root", type=Path, default=None)
    ap.add_argument("--gt-pattern", default="{case}/label.mha")
    ap.add_argument("--pred-pattern", default="{case}.mha")
    ap.add_argument(
        "--eval-mode",
        choices=("strict-official", "local-composite"),
        default="strict-official",
        help="strict-official requires a 2026 official evaluator; local-composite is for internal proxy exploration only",
    )
    args = ap.parse_args()

    case_list = None
    if args.cases:
        case_list = [l.strip() for l in args.cases.read_text().splitlines() if l.strip()]
    cfg = HarnessConfig(data_root=args.data_root, pred_root=args.pred_root,
                        out_dir=args.out_dir, cases=case_list,
                        gt_pattern=args.gt_pattern, pred_pattern=args.pred_pattern,
                        eval_mode=args.eval_mode)

    if args.validate_ref:
        assert args.public5_gt_root and args.public5_pred_root, \
            "--validate-ref 에는 --public5-gt-root / --public5-pred-root 필요"
        v = validate_against_reference(args.validate_ref, args.public5_gt_root,
                                       args.public5_pred_root, cfg)
        print(json.dumps(v, indent=2, ensure_ascii=False))
        if not v["unified"]:
            print("\n[!] harness != 공식 eval. 공식 평가기 버전을 동기화하세요(일원화 미완).")
        else:
            print("\n[ok] harness == 공식 eval 재현 확인. 내부 숫자를 신뢰해도 됨.")
        return

    report = run(cfg, sentinels=DEFAULT_SENTINELS, bonewise=not args.no_bonewise)

    if args.compare_to:
        base = json.loads(args.compare_to.read_text())
        cmp = compare_runs(base["aggregates"], report["aggregates"], cfg)
        (cfg.out_dir / "compare.json").write_text(json.dumps(cmp, indent=2, ensure_ascii=False))
        print("\n=== compare vs baseline ===")
        print(json.dumps(cmp["verdict"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()