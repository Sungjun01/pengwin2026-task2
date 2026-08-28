"""Sweep PIVOT-Form v0 hyperparameters on held-out 7-class preds (CPU-only).

Writes CSV comparing exact-count rate vs legacy postproc for each (tau_intact, theta_merge).

Usage:
    python3 scripts/sweep_pivot_form_params.py \\
        --pred_dir predictions/heldout_d015_ep999 \\
        --out_csv progress/pivot_form_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.measure_border_core_recovery import (
    ANATOMY,
    GT_INSTANCE,
    count_gt,
    count_postproc,
)
from scripts.pivot_form_postproc import FormConfig, assemble_pelvic_instances
from scripts.postproc_fragment_graph_merge import D015_ANATOMY, ANATOMY_TO_RANGE


def exact_count_instance_dir(
    inst_dir: Path,
    gt_dir: Path,
    min_volume_mm3: float,
) -> tuple[int, int, int]:
    """Return (n_exact, n_total, n_over)."""
    n_exact = n_total = n_over = 0
    for p in sorted(inst_dir.glob("*.nii.gz")):
        cid = p.name.split(".")[0].replace("PENGWIN_", "")
        gt_path = gt_dir / f"PENGWIN_{cid}.nii.gz"
        if not gt_path.exists():
            continue
        inst = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        img = sitk.ReadImage(str(p))
        vox_mm3 = float(np.prod(img.GetSpacing()))

        for name, (_, _, (lo, hi)) in ANATOMY.items():
            gt = count_gt(ginst, vox_mm3, lo, hi, min_volume_mm3)
            pred = 0
            for fid in range(lo, hi + 1):
                if int((inst == fid).sum()) * vox_mm3 >= min_volume_mm3:
                    pred += 1
            if gt == 0 and pred == 0:
                continue
            n_total += 1
            if gt == pred:
                n_exact += 1
            elif pred > gt:
                n_over += 1
    return n_exact, n_total, n_over


def run_form_on_dir(pred_dir: Path, form_cfg: FormConfig | None) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="form_sweep_"))
    for p in sorted(pred_dir.glob("*.nii.gz")):
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img).astype(np.int16)
        sp = img.GetSpacing()
        spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        vox = float(np.prod(sp))
        inst = assemble_pelvic_instances(
            arr, vox, D015_ANATOMY, ANATOMY_TO_RANGE,
            form_cfg=form_cfg, spacing_zyx=spacing_zyx,
        )
        out_img = sitk.GetImageFromArray(inst.astype(np.uint16))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(tmp / p.name))
    return tmp


def legacy_exact_from_7class(pred_dir: Path, gt_dir: Path, min_vol: float) -> tuple[int, int, int]:
    n_exact = n_total = n_over = 0
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.name.split(".")[0].replace("PENGWIN_", "")
        parr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path := gt_dir / f"PENGWIN_{cid}.nii.gz")))
        if not gt_path.exists():
            continue
        img = sitk.ReadImage(str(p))
        vox_mm3 = float(np.prod(img.GetSpacing()))
        for name, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
            gt = count_gt(ginst, vox_mm3, lo, hi, min_vol)
            pr = count_postproc(parr == core_cls, parr == border_cls, vox_mm3, min_vol)
            if gt == 0 and pr == 0:
                continue
            n_total += 1
            if gt == pr:
                n_exact += 1
            elif pr > gt:
                n_over += 1
    return n_exact, n_total, n_over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--gt_dir", type=Path, default=GT_INSTANCE)
    ap.add_argument("--out_csv", type=Path, default=_ROOT / "progress" / "pivot_form_sweep.csv")
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument(
        "--tau_values", type=str, default="0.05,0.08,0.10,0.12",
        help="comma-separated tau_intact values",
    )
    ap.add_argument(
        "--theta_values", type=str, default="0.25,0.35,0.45,0.55",
        help="comma-separated theta_merge values",
    )
    args = ap.parse_args()

    taus = [float(x) for x in args.tau_values.split(",")]
    thetas = [float(x) for x in args.theta_values.split(",")]

    le, lt, lo = legacy_exact_from_7class(args.pred_dir, args.gt_dir, args.min_volume_mm3)
    rows = [{
        "tau_intact": "legacy_7cl",
        "theta_merge": "",
        "n_exact": le,
        "n_total": lt,
        "pct_exact": f"{100 * le / max(lt, 1):.1f}",
        "n_over": lo,
    }]

    import shutil

    for tau in taus:
        for theta in thetas:
            cfg = FormConfig(
                min_volume_mm3=args.min_volume_mm3,
                tau_intact=tau,
                theta_merge=theta,
            )
            tmp = run_form_on_dir(args.pred_dir, cfg)
            ex, tot, ov = exact_count_instance_dir(tmp, args.gt_dir, args.min_volume_mm3)
            shutil.rmtree(tmp, ignore_errors=True)
            rows.append({
                "tau_intact": tau,
                "theta_merge": theta,
                "n_exact": ex,
                "n_total": tot,
                "pct_exact": f"{100 * ex / max(tot, 1):.1f}",
                "n_over": ov,
            })
            print(f"tau={tau} theta={theta} -> {ex}/{tot} exact, over={ov}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
