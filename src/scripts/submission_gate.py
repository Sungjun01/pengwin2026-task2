"""Submission gate for Task 1 candidates against the locked v18 baseline.

Input JSON shape:
{
  "baseline": {"003": {"ins_f1_score": ...}, "004": {...}, ...},
  "candidate": {"003": {...}, "004": {...}, ...},
  "runtime_s": 540.0
}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROXY_CASES = ("002", "004", "005")
PELVIC_CASE = "001"
BOUNDARY_CASE = "004"
SACRUM_CASE = "003"


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]
    notes: list[str]


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise KeyError(f"missing metric {key}")
    return float(value)


def _case(payload: dict[str, Any], section: str, case_id: str) -> dict[str, Any]:
    try:
        row = payload[section][case_id]
    except KeyError as exc:
        raise KeyError(f"missing {section}.{case_id}") from exc
    if not isinstance(row, dict):
        raise TypeError(f"{section}.{case_id} must be an object")
    return row


def evaluate_submission_gate(
    payload: dict[str, Any],
    *,
    max_runtime_s: float = 600.0,
    proxy_min_ins_f1: float = 0.999,
    min_004_hd95_gain_mm: float = 0.1,
) -> GateResult:
    failures: list[str] = []
    notes: list[str] = []

    for case_id in PROXY_CASES:
        base = _case(payload, "baseline", case_id)
        cand = _case(payload, "candidate", case_id)
        cand_f1 = _metric(cand, "ins_f1_score")
        base_f1 = _metric(base, "ins_f1_score")
        if cand_f1 < max(proxy_min_ins_f1, base_f1):
            failures.append(
                f"proxy {case_id}: ins_f1 {cand_f1:.3f} < required {max(proxy_min_ins_f1, base_f1):.3f}"
            )
        for key in ("merge_error_count", "split_error_count"):
            if _metric(cand, key) > _metric(base, key):
                failures.append(f"proxy {case_id}: {key} regressed")
        if _metric(cand, "topology_consistency") < _metric(base, "topology_consistency"):
            failures.append(f"proxy {case_id}: topology_consistency regressed")

    base004 = _case(payload, "baseline", BOUNDARY_CASE)
    cand004 = _case(payload, "candidate", BOUNDARY_CASE)
    if _metric(cand004, "fracture_hd95") > _metric(base004, "fracture_hd95") - min_004_hd95_gain_mm:
        failures.append("004 boundary: HD95 did not improve enough")
    if _metric(cand004, "fracture_assd") > _metric(base004, "fracture_assd"):
        failures.append("004 boundary: ASSD regressed")
    if _metric(cand004, "fracture_local_dice") < _metric(base004, "fracture_local_dice"):
        failures.append("004 boundary: local_dice regressed")

    base003 = _case(payload, "baseline", SACRUM_CASE)
    cand003 = _case(payload, "candidate", SACRUM_CASE)
    sacrum_f1_gain = _metric(cand003, "ins_f1_score") > _metric(base003, "ins_f1_score")
    sacrum_hd95_gain = _metric(cand003, "fracture_hd95") < _metric(base003, "fracture_hd95")
    if not (sacrum_f1_gain or sacrum_hd95_gain):
        failures.append("003 sacrum: neither ins_f1 nor HD95 improved")
    if _metric(cand003, "merge_error_count") > _metric(base003, "merge_error_count"):
        failures.append("003 sacrum: merge_error_count regressed")

    if PELVIC_CASE in payload.get("baseline", {}) and PELVIC_CASE in payload.get("candidate", {}):
        base001 = _case(payload, "baseline", PELVIC_CASE)
        cand001 = _case(payload, "candidate", PELVIC_CASE)
        precision_gain = _metric(cand001, "ins_precision") > _metric(base001, "ins_precision")
        split_gain = _metric(cand001, "split_error_count") < _metric(base001, "split_error_count")
        f1_gain = _metric(cand001, "ins_f1_score") > _metric(base001, "ins_f1_score")
        if precision_gain or split_gain or f1_gain:
            notes.append("001 improvement: precision/split/f1 moved in candidate direction")
        if _metric(cand001, "merge_error_count") > _metric(base001, "merge_error_count"):
            failures.append("001 pelvic: merge_error_count regressed")

    surface_refine = payload.get("surface_refine")
    if surface_refine is not None:
        before = [int(x) for x in surface_refine.get("ids_before", [])]
        after = [int(x) for x in surface_refine.get("ids_after", [])]
        if before != after:
            failures.append(f"surface refine: instance IDs changed before={before} after={after}")

    runtime_s = float(payload.get("runtime_s", max_runtime_s + 1.0))
    if runtime_s > max_runtime_s:
        failures.append(f"runtime: {runtime_s:.1f}s > {max_runtime_s:.1f}s")

    return GateResult(passed=not failures, failures=failures, notes=notes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1 candidate submission gate")
    parser.add_argument("metrics_json", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.metrics_json.read_text())
    result = evaluate_submission_gate(payload)
    if result.passed:
        print("SUBMISSION_GATE_PASS")
        for note in result.notes:
            print(f"- note: {note}")
        return 0
    print("SUBMISSION_GATE_FAIL")
    for failure in result.failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
