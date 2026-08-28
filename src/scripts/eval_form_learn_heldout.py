"""Evaluate Tier B postproc modes on held-out 6 (§4A.8.1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.form_learn_postproc import assemble_pelvic_instances_mode, load_merge_weights
from scripts.measure_border_core_recovery import ANATOMY, count_gt

HELDOUT = ("003", "011", "017", "018", "025", "034")
GT_INSTANCE = _ROOT / "predictions" / "gt_instance"


def count_inst_fragments(
    inst: np.ndarray, lo: int, hi: int, voxel_mm3: float, min_mm3: float
) -> int:
    min_vox = int(np.ceil(min_mm3 / voxel_mm3))
    n = 0
    for fid in range(lo, hi + 1):
        if int((inst == fid).sum()) >= min_vox:
            n += 1
    return n


def eval_mode(
    pred_dir: Path,
    mode: str,
    min_volume_mm3: float,
    merge_weights: dict | None,
    theta_merge: float,
) -> dict:
    exact = total = 0
    rows = []
    for case_id in HELDOUT:
        p = pred_dir / f"PENGWIN_{case_id}.nii.gz"
        g = GT_INSTANCE / f"PENGWIN_{case_id}.nii.gz"
        img = sitk.ReadImage(str(p))
        sem = sitk.GetArrayFromImage(img).astype(np.int16)
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(g)))
        sp = img.GetSpacing()
        voxel = float(sp[0] * sp[1] * sp[2])
        szyx = (float(sp[2]), float(sp[1]), float(sp[0]))

        inst = assemble_pelvic_instances_mode(
            sem,
            voxel,
            mode=mode,
            merge_weights=merge_weights,
            spacing_zyx=szyx,
            min_volume_mm3=min_volume_mm3,
            theta_merge=theta_merge,
        )

        for anatomy, (_, _, (lo, hi)) in ANATOMY.items():
            gt_n = count_gt(ginst, voxel, lo, hi, min_volume_mm3)
            if gt_n == 0:
                continue
            total += 1
            pred_n = count_inst_fragments(inst, lo, hi, voxel, min_volume_mm3)
            ok = pred_n == gt_n
            exact += int(ok)
            rows.append(
                {
                    "case_id": f"PENGWIN_{case_id}",
                    "anatomy": anatomy,
                    "n_gt": gt_n,
                    "n_pred": pred_n,
                    "exact": ok,
                }
            )
    return {"exact": exact, "total": total, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, default=_ROOT / "predictions" / "heldout_d015_ep999_foldall")
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument("--merge_model", type=Path, default=_ROOT / "progress" / "form_learn_merge.json")
    ap.add_argument("--theta", type=float, default=0.35)
    ap.add_argument("--out_json", type=Path, default=_ROOT / "progress" / "form_learn_heldout_eval.json")
    args = ap.parse_args()

    results = {}
    for mode in ("legacy", "form_intact", "keep_all", "hybrid", "learned"):
        mw = None
        if mode == "learned":
            if not args.merge_model.exists():
                print(f"[skip] learned — no model at {args.merge_model}")
                continue
            mw = load_merge_weights(args.merge_model)
        r = eval_mode(args.pred_dir, mode, args.min_volume_mm3, mw, args.theta)
        results[mode] = {"exact": r["exact"], "total": r["total"]}
        print(f"[{mode}] {r['exact']}/{r['total']} exact")

    best = max(results.items(), key=lambda x: x[1]["exact"])
    summary = {"pred_dir": str(args.pred_dir), "modes": results, "best_mode": best[0]}
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"best: {best[0]} ({best[1]['exact']}/{best[1]['total']})")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
