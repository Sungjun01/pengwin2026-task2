"""Border-core prediction -> fragment recovery measurement.

Takes a 7-class pelvic border-core prediction dir and the GT instance dir, and
for each case/anatomy counts how many fragments are recovered vs ground truth.

Recovery = take the CORE class voxels of an anatomy, 26-connectivity connected
components, drop components below --min_volume_mm3, count what remains. This is
the metric that actually decides the tuning grid (border *dice* is only a proxy).

7-class pred encoding (pelvic D012/D098/D099):
    0 bg
    1 sacrum_core   2 sacrum_border
    3 leftHip_core  4 leftHip_border
    5 rightHip_core 6 rightHip_border

GT instance encoding (PENGWIN): 1-50 sacrum, 51-100 leftHip, 101-150 rightHip.

용법:
    python3 scripts/measure_border_core_recovery.py \\
        --pred_dir <7class preds> --label A --min_volume_mm3 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label

GT_INSTANCE = Path(
    "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance"
)

# anatomy -> (core class, border class, GT instance id range)
ANATOMY = {
    "sacrum": (1, 2, (1, 50)),
    "leftHip": (3, 4, (51, 100)),
    "rightHip": (5, 6, (101, 150)),
}
CORE_STRUCT = generate_binary_structure(3, 3)  # 26-connectivity


def _cc_sizes(labels: np.ndarray, n: int) -> np.ndarray:
    """Voxel counts per CC id 1..n (index 0 unused)."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]


def count_components(mask: np.ndarray, voxel_mm3: float, min_mm3: float) -> int:
    if not mask.any():
        return 0
    labels, n = cc_label(mask, structure=CORE_STRUCT)
    sizes = _cc_sizes(labels, n)
    min_vox = int(np.ceil(min_mm3 / voxel_mm3))
    return int((sizes >= min_vox).sum())


def count_postproc(core_mask: np.ndarray, border_mask: np.ndarray,
                   voxel_mm3: float, min_mm3: float) -> int:
    """Border-adjacency post-proc (Task 5 §5.1): a core CC is a real fragment iff it
    is seam-bounded (a border voxel within 2 voxels) OR the largest in this anatomy."""
    if not core_mask.any():
        return 0
    labels, n = cc_label(core_mask, structure=CORE_STRUCT)
    sizes = _cc_sizes(labels, n)
    min_vox = int(np.ceil(min_mm3 / voxel_mm3))
    kept_ids = np.flatnonzero(sizes >= min_vox) + 1
    if kept_ids.size == 0:
        return 0
    largest = int(kept_ids[np.argmax(sizes[kept_ids - 1])])
    border_near = binary_dilation(border_mask, structure=CORE_STRUCT, iterations=2)
    kept = 0
    for c in kept_ids:
        if c == largest or np.any(border_near & (labels == c)):
            kept += 1
    return kept


def count_gt(inst: np.ndarray, voxel_mm3: float, lo: int, hi: int, min_mm3: float) -> int:
    kept = 0
    for fid in range(lo, hi + 1):
        vox = int((inst == fid).sum())
        if vox > 0 and vox * voxel_mm3 >= min_mm3:
            kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--label", default="?", help="config label for the table header")
    ap.add_argument("--min_volume_mm3", type=float, default=1000.0)
    ap.add_argument("--postproc", action="store_true",
                    help="apply border-adjacency post-proc (Task 5 §5.1) instead of raw core CC")
    a = ap.parse_args()

    preds = sorted(a.pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"no predictions under {a.pred_dir}")

    mode = "border-adjacency post-proc" if a.postproc else "raw core CC"
    print(f"\n=== config {a.label}  (min_volume={a.min_volume_mm3:.0f} mm³, {mode}) ===")
    print(f"{'case':>10} | {'anatomy':>8} | {'GT':>3} -> {'pred':>4} | verdict")
    n_exact = n_total = 0
    per_anat = {name: [0, 0, 0, 0] for name in ANATOMY}  # [exact, total, over, under]
    for p in preds:
        cid = p.name.split(".")[0].replace("PENGWIN_", "")
        pimg = sitk.ReadImage(str(p))
        parr = sitk.GetArrayFromImage(pimg)
        sx, sy, sz = pimg.GetSpacing()
        voxel_mm3 = float(sx * sy * sz)

        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(GT_INSTANCE / f"PENGWIN_{cid}.nii.gz")))

        for name, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
            gt = count_gt(ginst, voxel_mm3, lo, hi, a.min_volume_mm3)
            if a.postproc:
                pr = count_postproc(parr == core_cls, parr == border_cls, voxel_mm3, a.min_volume_mm3)
            else:
                pr = count_components(parr == core_cls, voxel_mm3, a.min_volume_mm3)
            if gt == 0 and pr == 0:
                continue
            n_total += 1
            per_anat[name][1] += 1
            if gt == pr:
                verdict = "exact ✓"
                n_exact += 1
                per_anat[name][0] += 1
            elif pr > gt:
                verdict = f"over-seg +{pr - gt}"
                per_anat[name][2] += 1
            else:
                verdict = f"under-seg {pr - gt}"
                per_anat[name][3] += 1
            print(f"PENGWIN_{cid:>3} | {name:>8} | {gt:>3} -> {pr:>4} | {verdict}")
    print(f"\n--- per-anatomy breakdown ---")
    for name, (ex, tot, ov, un) in per_anat.items():
        if tot:
            print(f"  {name:>8}: {ex}/{tot} exact ({100*ex/tot:.0f}%)  | over-seg {ov}, under-seg {un}")
    print(f"  {'TOTAL':>8}: {n_exact}/{n_total} exact ({100*n_exact/max(n_total,1):.0f}%)")


if __name__ == "__main__":
    main()
