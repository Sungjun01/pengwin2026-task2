"""Count exact fragment-match rate from PENGWIN instance label maps (not 7-class)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.measure_border_core_recovery import ANATOMY, GT_INSTANCE, count_gt


def exact_stats(inst_dir: Path, gt_dir: Path, min_vol: float) -> dict:
    n_exact = n_total = n_over = 0
    per_anat = {k: [0, 0, 0] for k in ANATOMY}
    for p in sorted(inst_dir.glob("*.nii.gz")):
        cid = p.name.split(".")[0].replace("PENGWIN_", "")
        gt_path = gt_dir / f"PENGWIN_{cid}.nii.gz"
        if not gt_path.exists():
            continue
        inst = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        vox = float(sitk.ReadImage(str(p)).GetSpacing()[0]
                    * sitk.ReadImage(str(p)).GetSpacing()[1]
                    * sitk.ReadImage(str(p)).GetSpacing()[2])
        for name, (_, _, (lo, hi)) in ANATOMY.items():
            gt = count_gt(ginst, vox, lo, hi, min_vol)
            pred = sum(
                1
                for fid in range(lo, hi + 1)
                if int((inst == fid).sum()) * vox >= min_vol
            )
            if gt == 0 and pred == 0:
                continue
            n_total += 1
            per_anat[name][1] += 1
            if gt == pred:
                n_exact += 1
                per_anat[name][0] += 1
            elif pred > gt:
                n_over += 1
                per_anat[name][2] += 1
    return {
        "n_exact": n_exact,
        "n_total": n_total,
        "n_over": n_over,
        "per_anat": per_anat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_dir", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--gt_dir", type=Path, default=GT_INSTANCE)
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    args = ap.parse_args()
    s = exact_stats(args.inst_dir, args.gt_dir, args.min_volume_mm3)
    tag = args.label or args.inst_dir.name
    print(f"\n=== {tag} (min_vol={args.min_volume_mm3}) ===")
    print(f"exact: {s['n_exact']}/{s['n_total']} ({100*s['n_exact']/max(s['n_total'],1):.0f}%)  over={s['n_over']}")
    for name, (ex, tot, ov) in s["per_anat"].items():
        if tot:
            print(f"  {name:>8}: {ex}/{tot} exact  over={ov}")


if __name__ == "__main__":
    main()
