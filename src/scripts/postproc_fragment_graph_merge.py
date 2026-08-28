"""Batch 7-class border-core predictions → PENGWIN instance maps (PIVOT-Form v0).

Usage:
    python3 scripts/postproc_fragment_graph_merge.py \\
        --pred_dir predictions/heldout_d015_ep999 \\
        --out_dir predictions/heldout_d015_ep999_form_v0 \\
        --tau_intact 0.08 --theta_merge 0.35

Then compare:
    python3 scripts/audit_fragment_formation.py --pred_dir <7class> ...
    python3 scripts/measure_border_core_recovery.py --pred_dir <instance> --postproc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pivot_form_postproc import FormConfig, assemble_pelvic_instances

D015_ANATOMY = {
    "sacrum": (1, 2),
    "leftHip": (3, 4),
    "rightHip": (5, 6),
}
ANATOMY_TO_RANGE = {
    "sacrum": (1, 50),
    "leftHip": (51, 100),
    "rightHip": (101, 150),
}


def process_case(
    pred_path: Path,
    out_path: Path,
    form_cfg: FormConfig,
) -> dict:
    img = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayFromImage(img).astype(np.int16)
    sp = img.GetSpacing()
    spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    voxel_mm3 = spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]

    inst = assemble_pelvic_instances(
        arr,
        voxel_mm3,
        D015_ANATOMY,
        ANATOMY_TO_RANGE,
        form_cfg=form_cfg,
        spacing_zyx=spacing_zyx,
    )

    out_img = sitk.GetImageFromArray(inst.astype(np.uint16))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path), useCompression=True)

    return {"max_id": int(inst.max()), "n_sac": int(np.max(inst // 1 * (inst <= 50))),}


def main() -> None:
    ap = argparse.ArgumentParser(description="PIVOT-Form v0 batch post-processing")
    ap.add_argument("--pred_dir", required=True, type=Path, help="7-class nii.gz directory")
    ap.add_argument("--out_dir", required=True, type=Path, help="instance output directory")
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument("--tau_intact", type=float, default=0.08)
    ap.add_argument("--theta_merge", type=float, default=0.35)
    ap.add_argument("--d_max_mm", type=float, default=80.0)
    ap.add_argument("--legacy", action="store_true", help="disable Form (legacy adjacency only)")
    args = ap.parse_args()

    preds = sorted(args.pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"no .nii.gz in {args.pred_dir}")

    form_cfg = None if args.legacy else FormConfig(
        min_volume_mm3=args.min_volume_mm3,
        tau_intact=args.tau_intact,
        theta_merge=args.theta_merge,
        d_max_mm=args.d_max_mm,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mode = "legacy" if args.legacy else "pivot-form-v0"
    print(f"[pivot-form] mode={mode}  n_cases={len(preds)}  out={args.out_dir}")

    for p in preds:
        out_path = args.out_dir / p.name
        info = process_case(p, out_path, form_cfg)
        print(f"  {p.name}  max_id={info['max_id']}")


if __name__ == "__main__":
    main()
