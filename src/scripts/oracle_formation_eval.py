"""Tier 0 — Oracle BC formation decomposition (§4A.8.4).

GT D015 7-class border-core → legacy (and optional Form) postproc → fragment counts
vs GT instance. Classifies failures: rule-error vs representation-degenerate.

Usage:
    python3 scripts/oracle_formation_eval.py
    python3 scripts/oracle_formation_eval.py --pred_dir predictions/heldout_d015_ep999_foldall
    python3 scripts/oracle_formation_eval.py --cases 003 011 --out_csv progress/tier0_oracle.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.measure_border_core_recovery import (
    ANATOMY,
    count_components,
    count_gt,
    count_postproc,
)
from scripts.pivot_form_postproc import FormConfig, assemble_pelvic_instances

HELDOUT_CASES = ("003", "011", "017", "018", "025", "034")
DEFAULT_LABELS = Path("/home/schoaq/taehee/nnUNet_data/raw/Dataset015_PENGWIN_BorderCore_OrientFixed/labelsTr")
DEFAULT_GT = _ROOT / "predictions" / "gt_instance"


def classify_failure(
    n_gt: int,
    n_core_cc: int,
    n_postproc: int,
) -> str:
    if n_gt == 0 and n_postproc == 0:
        return "ok"
    if n_postproc == n_gt:
        return "ok"
    if n_gt >= 2 and n_core_cc <= 1:
        return "representation-degenerate"
    if n_core_cc >= 2:
        return "rule-error"
    if n_gt >= 1 and n_core_cc == 0:
        return "representation-degenerate"
    return "unknown"


def eval_semantic(
    sem: np.ndarray,
    ginst: np.ndarray,
    voxel_mm3: float,
    min_volume_mm3: float,
    use_form: bool,
) -> dict[str, dict]:
    form_cfg = None
    if use_form:
        form_cfg = FormConfig(min_volume_mm3=min_volume_mm3)

    out: dict[str, dict] = {}
    for name, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
        core = sem == core_cls
        border = sem == border_cls
        n_gt = count_gt(ginst, voxel_mm3, lo, hi, min_volume_mm3)
        n_core = count_components(core, voxel_mm3, min_volume_mm3)
        n_legacy = count_postproc(core, border, voxel_mm3, min_volume_mm3)

        if use_form:
            inst = assemble_pelvic_instances(
                sem.astype(np.int16),
                voxel_mm3,
                {name: (core_cls, border_cls)},
                {name: (lo, hi)},
                form_cfg=form_cfg,
                spacing_zyx=(1.0, 1.0, 1.0),
            )
            n_form = 0
            for fid in range(lo, hi + 1):
                vox = int((inst == fid).sum())
                if vox > 0 and vox * voxel_mm3 >= min_volume_mm3:
                    n_form += 1
            n_post = n_form
        else:
            n_post = n_legacy

        mode = classify_failure(n_gt, n_core, n_post)
        out[name] = {
            "n_gt": n_gt,
            "n_core_cc": n_core,
            "n_legacy": n_legacy,
            "n_postproc": n_post,
            "failure_mode": mode,
            "exact": (n_gt == 0 and n_post == 0) or (n_post == n_gt),
        }
    return out


def oracle_merge_ceiling_per_anatomy(
    ginst: np.ndarray,
    sem: np.ndarray,
    anatomy: str,
    voxel_mm3: float,
    min_volume_mm3: float,
) -> int:
    """Upper bound: one core CC per GT instance (GT instance masks ∩ GT BC core)."""
    core_cls, border_cls, (lo, hi) = ANATOMY[anatomy]
    core_sem = sem == core_cls
    min_vox = int(np.ceil(min_volume_mm3 / voxel_mm3))
    n = 0
    for fid in np.unique(ginst):
        if lo <= int(fid) <= hi:
            vox = int(((ginst == fid) & core_sem).sum())
            if vox >= min_vox:
                n += 1
    return n


def process_case(
    case_id: str,
    labels_dir: Path,
    gt_dir: Path,
    pred_path: Path | None,
    min_volume_mm3: float,
    use_form: bool,
) -> list[dict]:
    cid = f"PENGWIN_{case_id}"
    lab_path = labels_dir / f"{cid}.nii.gz"
    gt_path = gt_dir / f"{cid}.nii.gz"
    if not lab_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"missing {lab_path} or {gt_path}")

    lab_img = sitk.ReadImage(str(lab_path))
    sem = sitk.GetArrayFromImage(lab_img).astype(np.int16)
    ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    sp = lab_img.GetSpacing()
    voxel_mm3 = float(sp[0] * sp[1] * sp[2])

    oracle = eval_semantic(sem, ginst, voxel_mm3, min_volume_mm3, use_form)
    pred_oracle = None
    if pred_path and pred_path.exists():
        pimg = sitk.ReadImage(str(pred_path))
        parr = sitk.GetArrayFromImage(pimg).astype(np.int16)
        pred_oracle = eval_semantic(parr, ginst, voxel_mm3, min_volume_mm3, use_form=False)

    rows: list[dict] = []
    for name in ANATOMY:
        o = oracle[name]
        n_ceiling = oracle_merge_ceiling_per_anatomy(ginst, sem, name, voxel_mm3, min_volume_mm3)
        row = {
            "case_id": cid,
            "anatomy": name,
            "n_gt": o["n_gt"],
            "n_core_cc_oracle": o["n_core_cc"],
            "n_legacy_oracle": o["n_legacy"],
            "n_postproc_oracle": o["n_postproc"],
            "oracle_merge_ceiling": n_ceiling,
            "failure_mode": o["failure_mode"],
            "oracle_exact": o["exact"],
        }
        if pred_oracle:
            p = pred_oracle[name]
            row["n_postproc_pred"] = p["n_postproc"]
            row["pred_exact"] = p["exact"]
            row["pred_failure_mode"] = p["failure_mode"]
        rows.append(row)
    return rows


def summarize_oracle(rows: list[dict]) -> dict:
    slots = [r for r in rows if r["n_gt"] > 0 or r["n_postproc_oracle"] > 0]
    by_mode: dict[str, int] = {}
    for r in slots:
        if not r["oracle_exact"]:
            m = r["failure_mode"]
            by_mode[m] = by_mode.get(m, 0) + 1
    return {
        "exact": sum(1 for r in slots if r["oracle_exact"]),
        "total": len(slots),
        "failures_by_mode": by_mode,
    }


def summarize_pred(rows: list[dict]) -> dict:
    slots = [r for r in rows if r["n_gt"] > 0 or r.get("n_postproc_pred", 0) > 0]
    by_mode: dict[str, int] = {}
    for r in slots:
        if not r.get("pred_exact", True):
            m = r.get("pred_failure_mode", "unknown")
            by_mode[m] = by_mode.get(m, 0) + 1
    return {
        "exact": sum(1 for r in slots if r.get("pred_exact")),
        "total": len(slots),
        "failures_by_mode": by_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier 0 oracle formation eval (§4A.8.4)")
    ap.add_argument("--labels_dir", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--gt_dir", type=Path, default=DEFAULT_GT)
    ap.add_argument("--pred_dir", type=Path, default=None, help="7-class preds for pred_* columns")
    ap.add_argument("--cases", nargs="*", default=None, help="case nums; default held-out 6")
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument("--form", action="store_true", help="use Form v0.2 for oracle postproc (else legacy)")
    ap.add_argument("--out_csv", type=Path, default=_ROOT / "progress" / "tier0_oracle_formation.csv")
    ap.add_argument("--out_json", type=Path, default=_ROOT / "progress" / "tier0_oracle_formation_summary.json")
    args = ap.parse_args()

    cases = args.cases or list(HELDOUT_CASES)
    all_rows: list[dict] = []

    for case_id in cases:
        print(f"[tier0] case {case_id} ...", flush=True)
        pred_path = None
        if args.pred_dir:
            pred_path = args.pred_dir / f"PENGWIN_{case_id}.nii.gz"
        all_rows.extend(
            process_case(
                case_id,
                args.labels_dir,
                args.gt_dir,
                pred_path,
                args.min_volume_mm3,
                args.form,
            )
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    oracle_sum = summarize_oracle(all_rows)
    summary = {
        "cases": cases,
        "min_volume_mm3": args.min_volume_mm3,
        "postproc": "form_v0" if args.form else "legacy",
        "pred_dir": str(args.pred_dir) if args.pred_dir else None,
        "oracle_heldout_slots": oracle_sum,
    }
    if args.pred_dir:
        summary["pred_heldout_slots"] = summarize_pred(all_rows)

    # Routing recommendation (§4A.8.4)
    fail_modes = oracle_sum.get("failures_by_mode", {})
    n_degen = fail_modes.get("representation-degenerate", 0)
    n_rule = fail_modes.get("rule-error", 0)
    if n_degen > n_rule:
        route = "D016_first"
    elif n_rule >= n_degen and n_rule > 0:
        route = "Tier_B_Form_Learn"
    elif oracle_sum["exact"] == oracle_sum["total"]:
        route = "Tier_A_optional"
    else:
        route = "mixed_review"

    summary["recommended_route"] = route
    args.out_json.write_text(json.dumps(summary, indent=2))

    print(f"[tier0] cases={cases}  postproc={'form' if args.form else 'legacy'}")
    print(f"  oracle held-out slots: {oracle_sum['exact']}/{oracle_sum['total']} exact")
    if fail_modes:
        print(f"  oracle failures: {fail_modes}")
    if args.pred_dir and "pred_heldout_slots" in summary:
        ps = summary["pred_heldout_slots"]
        print(f"  pred  held-out slots: {ps['exact']}/{ps['total']} exact")
    print(f"  recommended_route: {route}")
    print(f"  wrote {args.out_csv}")
    print(f"  wrote {args.out_json}")

    # Per-case table
    print(f"\n{'case':>12} {'anatomy':>8} | GT -> oracle | mode")
    for r in all_rows:
        if r["n_gt"] == 0 and r["n_postproc_oracle"] == 0:
            continue
        ex = "OK" if r["oracle_exact"] else "MISS"
        print(
            f"{r['case_id']:>12} {r['anatomy']:>8} | "
            f"{r['n_gt']:>2} -> {r['n_postproc_oracle']:>2} (cc={r['n_core_cc_oracle']}) | "
            f"{r['failure_mode']:<26} {ex}"
        )


if __name__ == "__main__":
    main()
