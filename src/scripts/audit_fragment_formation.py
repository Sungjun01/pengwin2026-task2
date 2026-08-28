"""Per-case fragment formation audit (nnU-Net limitation K1/K2/K3/A4).

For each case and anatomy, records GT fragment count, raw core CC count, border-adjacency
postproc count, and border/core voxel ratio on the predicted core mask. Use to show that
seam prediction can be strong while instance cardinality fails (formation bottleneck).

Usage:
    python3 scripts/audit_fragment_formation.py \\
        --pred_dir predictions/heldout_d015_ep999 \\
        --out_csv progress/formation_audit_d015_ep999.csv

See docs/NNUNET_LIMITATIONS_PENGWIN.md §2 (experiments A4, K2, K3).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.measure_border_core_recovery import (
    ANATOMY,
    CORE_STRUCT,
    GT_INSTANCE,
    count_components,
    count_gt,
    count_postproc,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "progress" / "formation_audit.csv"


def border_core_ratio(core_mask: np.ndarray, border_mask: np.ndarray) -> float:
    n_core = int(core_mask.sum())
    if n_core == 0:
        return float("nan")
    n_border_on_core = int((core_mask & border_mask).sum())
    # Also count border voxels within 2-voxel dilation of core (seam band)
    band = binary_dilation(core_mask, structure=CORE_STRUCT, iterations=2) & border_mask
    n_band = int(band.sum())
    return n_band / n_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--gt_dir", type=Path, default=GT_INSTANCE)
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument("--out_csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--label", default="", help="config tag stored in each row")
    args = ap.parse_args()

    preds = sorted(args.pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"no predictions under {args.pred_dir}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "case_id",
        "anatomy",
        "n_gt",
        "n_pred_raw",
        "n_pred_postproc",
        "border_core_ratio",
        "delta_raw",
        "delta_postproc",
        "verdict_postproc",
    ]
    rows: list[dict] = []

    for p in preds:
        cid = p.name.split(".")[0].replace("PENGWIN_", "")
        pimg = sitk.ReadImage(str(p))
        parr = sitk.GetArrayFromImage(pimg)
        sx, sy, sz = pimg.GetSpacing()
        voxel_mm3 = float(sx * sy * sz)

        gt_path = args.gt_dir / f"PENGWIN_{cid}.nii.gz"
        if not gt_path.exists():
            raise SystemExit(f"missing GT {gt_path}")
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

        for name, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
            core_mask = parr == core_cls
            border_mask = parr == border_cls
            n_gt = count_gt(ginst, voxel_mm3, lo, hi, args.min_volume_mm3)
            n_raw = count_components(core_mask, voxel_mm3, args.min_volume_mm3)
            n_pp = count_postproc(core_mask, border_mask, voxel_mm3, args.min_volume_mm3)
            ratio = border_core_ratio(core_mask, border_mask)

            if n_gt == 0 and n_raw == 0 and n_pp == 0:
                continue

            d_pp = n_pp - n_gt
            if d_pp == 0:
                verdict = "exact"
            elif d_pp > 0:
                verdict = "over"
            else:
                verdict = "under"

            rows.append(
                {
                    "label": args.label or args.pred_dir.name,
                    "case_id": cid,
                    "anatomy": name,
                    "n_gt": n_gt,
                    "n_pred_raw": n_raw,
                    "n_pred_postproc": n_pp,
                    "border_core_ratio": f"{ratio:.4f}" if ratio == ratio else "",
                    "delta_raw": n_raw - n_gt,
                    "delta_postproc": d_pp,
                    "verdict_postproc": verdict,
                }
            )

    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_exact = sum(1 for r in rows if r["verdict_postproc"] == "exact")
    n_over = sum(1 for r in rows if r["verdict_postproc"] == "over")
    print(f"Wrote {n} rows -> {args.out_csv}")
    print(f"postproc: {n_exact}/{n} exact ({100 * n_exact / max(n, 1):.0f}%), over={n_over}, under={n - n_exact - n_over}")


if __name__ == "__main__":
    main()
