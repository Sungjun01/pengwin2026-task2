#!/usr/bin/env python3
"""Apply topology-preserving surface_refine to decoded instance predictions.

GPU-free Lever A (progress_report_103 §4) experiment driver:
  - reads decoded instance .mha (already in LPS-oriented voxel grid),
  - reorients the raw CT to LPS so HU aligns voxel-wise with the prediction,
  - applies refine_instance_surfaces (cannot merge/split/rename instances),
  - writes refined predictions to a new pred_root for the proxy harness.

The output pred_root is a drop-in replacement for the baseline decoded_instances
dir, so `scripts/test.py --compare-to <baseline report>` gives the A/B verdict.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.surface_refine import SurfaceRefineConfig, refine_instance_surfaces


def _lps_ct_array(ct_path: Path, ref: sitk.Image) -> np.ndarray:
    """Return CT HU as an array aligned voxel-wise with the prediction grid.

    Predictions are stored in DICOMOrient(LPS) order; reorient CT the same way
    and sanity-check that the resulting grid matches the prediction.
    """
    ct = sitk.DICOMOrient(sitk.ReadImage(str(ct_path)), "LPS")
    if ct.GetSize() != ref.GetSize():
        raise ValueError(f"CT size {ct.GetSize()} != pred size {ref.GetSize()} ({ct_path})")
    return sitk.GetArrayFromImage(ct).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True,
                    help="baseline decoded instance dir (PENGWIN_{case}.mha)")
    ap.add_argument("--ct-root", type=Path, required=True,
                    help="raw_data/PENGWIN_train ({case}/image.mha)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--trim-hu", type=float, default=50.0)
    ap.add_argument("--recover-hu", type=float, default=300.0)
    ap.add_argument("--max-iters", type=int, default=1)
    args = ap.parse_args()

    cfg = SurfaceRefineConfig(
        trim_hu_threshold=args.trim_hu,
        recover_hu_threshold=args.recover_hu,
        max_iters=args.max_iters,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)

    preds = sorted(args.pred_root.glob("PENGWIN_*.mha"))
    print(f"[surface-refine] {len(preds)} cases | trim<{args.trim_hu} recover>={args.recover_hu} iters={args.max_iters}")
    for p in preds:
        case = p.stem.replace("PENGWIN_", "")
        pred_img = sitk.ReadImage(str(p))
        labels = sitk.GetArrayFromImage(pred_img).astype(np.uint16)
        ct = _lps_ct_array(args.ct_root / case / "image.mha", pred_img)

        refined = refine_instance_surfaces(labels, ct, cfg)
        changed = int((refined != labels).sum())

        out_img = sitk.GetImageFromArray(refined.astype(np.uint16))
        out_img.CopyInformation(pred_img)
        sitk.WriteImage(out_img, str(args.out_root / p.name), useCompression=True)
        print(f"  {case}: changed {changed:>8d} voxels  (ids {len(np.unique(labels))-1}->{len(np.unique(refined))-1})")


if __name__ == "__main__":
    main()
