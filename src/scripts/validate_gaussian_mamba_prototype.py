"""Validation gate for the Gaussian-Mamba research arm.

This intentionally blocks submission integration until D004 fold_0 and OOF
radar evidence show the primitive representation is at least non-regressive.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_ANATOMY_BUCKETS = ("sacrum", "pelvic_ring", "acetabulum_hip", "femur")
RADAR_THREAT_SIGNALS = ("JAM_OVER_DECODE", "JAM_UNDER_MODEL", "CLUTTER")


@dataclass(frozen=True)
class GaussianMambaValidationResult:
    ok: bool
    can_integrate_submission: bool
    reasons: tuple[str, ...]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _threat_count(report: dict[str, Any], model_key: str, signal: str) -> int:
    radar = report.get("oof_radar", {})
    threats = radar.get(model_key, {})
    return int(threats.get(signal, 0))


def validate_gaussian_mamba_prototype(
    checkpoint_path: str | Path,
    primitive_npz_path: str | Path,
    validation_report_path: str | Path,
) -> GaussianMambaValidationResult:
    reasons: list[str] = []
    checkpoint = Path(checkpoint_path)
    primitives = Path(primitive_npz_path)
    report_path = Path(validation_report_path)
    if not checkpoint.exists():
        reasons.append(f"missing fold_0 checkpoint: {checkpoint}")
    if not primitives.exists():
        reasons.append(f"missing primitive artifact: {primitives}")
    if not report_path.exists():
        reasons.append(f"missing validation report: {report_path}")
        return GaussianMambaValidationResult(False, False, tuple(reasons))

    report = _load_json(report_path)
    d004 = report.get("d004_fold0", {})
    baseline_rq = float(d004.get("baseline_rq", -1.0))
    gm_rq = float(d004.get("gaussian_mamba_rq", -1.0))
    if gm_rq < baseline_rq:
        reasons.append(f"D004 fold_0 RQ regression: gaussian_mamba_rq={gm_rq} < baseline_rq={baseline_rq}")

    for signal in RADAR_THREAT_SIGNALS:
        before = _threat_count(report, "baseline_threats", signal)
        after = _threat_count(report, "gaussian_mamba_threats", signal)
        if after > before:
            reasons.append(f"OOF radar threat increased for {signal}: {after} > {before}")

    per_anatomy = report.get("per_anatomy", {})
    if not isinstance(per_anatomy, dict):
        per_anatomy = {}
    for anatomy in REQUIRED_ANATOMY_BUCKETS:
        metrics = per_anatomy.get(anatomy)
        if not isinstance(metrics, dict):
            reasons.append(f"missing per-anatomy metrics for {anatomy}")
            continue
        anatomy_baseline_rq = float(metrics.get("baseline_rq", -1.0))
        anatomy_gm_rq = float(metrics.get("gaussian_mamba_rq", -1.0))
        if anatomy_gm_rq < anatomy_baseline_rq:
            reasons.append(
                f"{anatomy} RQ regression: gaussian_mamba_rq={anatomy_gm_rq} "
                f"< baseline_rq={anatomy_baseline_rq}"
            )

    radar_per_anatomy = report.get("oof_radar", {}).get("per_anatomy", {})
    if not isinstance(radar_per_anatomy, dict):
        radar_per_anatomy = {}
    for anatomy in REQUIRED_ANATOMY_BUCKETS:
        radar_metrics = radar_per_anatomy.get(anatomy)
        if not isinstance(radar_metrics, dict):
            reasons.append(f"missing per-anatomy OOF radar threats for {anatomy}")
            continue
        baseline_threats = radar_metrics.get("baseline_threats", {})
        gm_threats = radar_metrics.get("gaussian_mamba_threats", {})
        for signal in RADAR_THREAT_SIGNALS:
            before = int(baseline_threats.get(signal, 0))
            after = int(gm_threats.get(signal, 0))
            if after > before:
                reasons.append(f"{anatomy} OOF radar threat increased for {signal}: {after} > {before}")

    ranking = report.get("ranking")
    if not isinstance(ranking, dict):
        reasons.append("missing Gaussian primitive ranking metrics")
    else:
        topk_recall = float(ranking.get("topk_fragment_recall", -1.0))
        if topk_recall < 1.0:
            reasons.append(f"topk_fragment_recall below required full GT coverage: {topk_recall} < 1.0")
        positive_rank = float(ranking.get("mean_rank_positive_gt", -1.0))
        background_rank = float(ranking.get("mean_rank_background", -1.0))
        if positive_rank <= background_rank:
            reasons.append(
                f"mean rank for positive GT must exceed background: "
                f"positive GT={positive_rank} <= background={background_rank}"
            )
        baseline_clutter_rate = float(ranking.get("baseline_clutter_rate", -1.0))
        gm_clutter_rate = float(ranking.get("gaussian_mamba_clutter_rate", -1.0))
        if gm_clutter_rate > baseline_clutter_rate:
            reasons.append(
                f"Gaussian primitive clutter rate regression: {gm_clutter_rate} > {baseline_clutter_rate}"
            )

        per_anatomy_ndcg = ranking.get("per_anatomy_ndcg", {})
        if not isinstance(per_anatomy_ndcg, dict):
            per_anatomy_ndcg = {}
        for anatomy in REQUIRED_ANATOMY_BUCKETS:
            ndcg = per_anatomy_ndcg.get(anatomy)
            if not isinstance(ndcg, dict):
                reasons.append(f"missing per-anatomy ranking NDCG for {anatomy}")
                continue
            baseline_ndcg = float(ndcg.get("baseline", -1.0))
            gm_ndcg = float(ndcg.get("gaussian_mamba", -1.0))
            if gm_ndcg < baseline_ndcg:
                reasons.append(f"{anatomy} ranking NDCG regression: {gm_ndcg} < {baseline_ndcg}")

    decoded = report.get("decoded_instances")
    if not isinstance(decoded, dict):
        reasons.append("missing decoded instance metrics")
    else:
        baseline_decoded_rq = float(decoded.get("baseline_rq", -1.0))
        gm_decoded_rq = float(decoded.get("gaussian_mamba_rq", -1.0))
        if gm_decoded_rq < baseline_decoded_rq:
            reasons.append(
                f"decoded instance RQ regression: gaussian_mamba_rq={gm_decoded_rq} "
                f"< baseline_rq={baseline_decoded_rq}"
            )
        baseline_count_error = float(decoded.get("baseline_fragment_count_error", -1.0))
        gm_count_error = float(decoded.get("gaussian_mamba_fragment_count_error", -1.0))
        if gm_count_error > baseline_count_error:
            reasons.append(
                f"decoded fragment count error regression: {gm_count_error} > {baseline_count_error}"
            )
        baseline_hd95 = float(decoded.get("baseline_hd95_mm", -1.0))
        gm_hd95 = float(decoded.get("gaussian_mamba_hd95_mm", -1.0))
        if gm_hd95 > baseline_hd95:
            reasons.append(f"decoded HD95 regression: {gm_hd95} > {baseline_hd95}")
        baseline_assd = float(decoded.get("baseline_assd_mm", -1.0))
        gm_assd = float(decoded.get("gaussian_mamba_assd_mm", -1.0))
        if gm_assd > baseline_assd:
            reasons.append(f"decoded ASSD regression: {gm_assd} > {baseline_assd}")
        hard_cases = decoded.get("hard_cases", {})
        if isinstance(hard_cases, dict):
            for case_id, case_metrics in sorted(hard_cases.items()):
                if not isinstance(case_metrics, dict):
                    reasons.append(f"hard case {case_id} decoded metrics are invalid")
                    continue
                case_baseline_rq = float(case_metrics.get("baseline_rq", -1.0))
                case_gm_rq = float(case_metrics.get("gaussian_mamba_rq", -1.0))
                if case_gm_rq < case_baseline_rq:
                    reasons.append(
                        f"hard case {case_id} decoded RQ regression: "
                        f"{case_gm_rq} < {case_baseline_rq}"
                    )
                case_baseline_count_error = float(case_metrics.get("baseline_fragment_count_error", -1.0))
                case_gm_count_error = float(case_metrics.get("gaussian_mamba_fragment_count_error", -1.0))
                if case_gm_count_error > case_baseline_count_error:
                    reasons.append(
                        f"hard case {case_id} fragment count error regression: "
                        f"{case_gm_count_error} > {case_baseline_count_error}"
                    )

    memory_gb = report.get("runtime", {}).get("public_femur_peak_memory_gb")
    if memory_gb is None:
        reasons.append("missing runtime memory measurement for one public femur CT")
    elif float(memory_gb) <= 0:
        reasons.append(f"invalid runtime memory measurement: {memory_gb}")

    ok = len(reasons) == 0
    return GaussianMambaValidationResult(ok=ok, can_integrate_submission=ok, reasons=tuple(reasons))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--primitives", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = validate_gaussian_mamba_prototype(args.checkpoint, args.primitives, args.report)
    if result.ok:
        print("Gaussian-Mamba prototype gate: PASS")
        print("Submission integration may be considered behind PIVOT_GAUSSIAN_MAMBA=1.")
        return 0
    print("Gaussian-Mamba prototype gate: BLOCKED")
    for reason in result.reasons:
        print(f"- {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
