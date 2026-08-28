"""Phase 0 로컬 스코어보드.

held-out border-core raw 예측(D015/D016, 7-class) 을 LPS 정합 후
border_core_to_instances (legacy) 로 instance(1-200) decode → gt_instance 와
official pengwin_eval(6-metric) 로 채점.

orientation trap 대응: 예측은 LPS, gt_instance 는 mixed(LPS/RAS) 이므로
**둘 다 DICOMOrient(LPS)** 후 비교한다 (안 하면 RAS case 점수 거짓).

용법:
  python -m scripts.eval_heldout_bordercore \
      --pred_dir predictions/heldout_d015_ep999_foldall \
      --gt_dir   predictions/gt_instance \
      --tag d015_mv500 --min_vol 500 --work /tmp/p0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode  # noqa: E402
from evaluation.pengwin_eval import evaluate_submission  # noqa: E402


def reorient_lps(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def decode_to_instances(semantic_lps: sitk.Image, min_vol: float,
                        mode: str = "legacy") -> np.ndarray:
    arr = sitk.GetArrayFromImage(semantic_lps)  # zyx
    sx, sy, sz = semantic_lps.GetSpacing()
    voxel_vol = float(sx * sy * sz)
    spacing_zyx = (sz, sy, sx)
    return assemble_pelvic_instances_mode(
        arr, voxel_vol, mode=mode, spacing_zyx=spacing_zyx,
        min_volume_mm3=min_vol,
    ).astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min_vol", type=float, default=500.0)
    ap.add_argument("--mode", default="legacy",
                    choices=["legacy", "keep_all", "hybrid", "form_intact"])
    ap.add_argument("--work", type=Path, default=Path("/tmp/pengwin_p0"))
    ap.add_argument("--min_cc_mm3", type=float, default=None,
                    help="공식 평가의 small-CC cleanup (≈1000). 미지정=끔.")
    args = ap.parse_args()

    pred_out = args.work / args.tag / "pred"
    gt_out = args.work / args.tag / "gt"
    pred_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in args.pred_dir.glob("PENGWIN_*.nii.gz"))
    print(f"[{args.tag}] {len(cases)} cases, min_vol={args.min_vol}")
    for p in cases:
        name = p.name
        gt_p = args.gt_dir / name
        if not gt_p.exists():
            print(f"  skip {name}: no GT")
            continue
        sem = reorient_lps(sitk.ReadImage(str(p)))
        inst = decode_to_instances(sem, args.min_vol, mode=args.mode)
        inst_img = sitk.GetImageFromArray(inst.astype(np.uint16))
        inst_img.CopyInformation(sem)
        sitk.WriteImage(inst_img, str(pred_out / name))

        gt_img = reorient_lps(sitk.ReadImage(str(gt_p)))
        sitk.WriteImage(gt_img, str(gt_out / name))

    out_json = args.work / f"{args.tag}_eval.json"
    res = evaluate_submission(pred_out, gt_out, out_json, matching_method="hungarian",
                              min_cc_mm3=args.min_cc_mm3)
    agg = res["aggregate"]
    print(f"  --- {args.tag} aggregate ({len(res['per_case'])} cases) ---")
    for k in ("IoU_F_mean", "LDSC_F_mean", "RQ_F_mean",
              "HD95_F_mm_mean", "ASSD_F_mm_mean",
              "IoU_A_mean", "HD95_A_mm_mean", "ASSD_A_mm_mean"):
        print(f"    {k:18s} {agg.get(k, float('nan')):.4f}")
    # per-case RQ/LDSC for weak-case spotting
    print("  per-case  RQ_F / LDSC_F / n_gt / n_pred:")
    for cid, c in res["per_case"].items():
        print(f"    {cid:16s} RQ={c.get('RQ_F', float('nan')):.3f} "
              f"LDSC={c.get('LDSC_F', float('nan')):.3f} "
              f"ngt={c.get('n_gt_fragments')} npred={c.get('n_pred_fragments')}")


if __name__ == "__main__":
    main()
