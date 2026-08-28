"""v22 selection 탐색: (decode mode × min_vol) 를 RQ_F(=ins_f1) 로만 빠르게 스윕.

HD95/ASSD(느림) 생략하고 RQ_F 만 — selection 튜닝의 표적 메트릭이 ins_f1 이므로.
공식 평가 정합: 채점 전 1cm³ small-CC cleanup 적용(remove_small_components).

held-out border-core 예측(7-class) + gt_instance 로 채점.
v18 기준선(legacy, min_vol=500) 대비 어떤 config 가 RQ_F 를 올리는지 표로.

용법: python -m scripts.decode_sweep --pred_dir predictions/d016_heldout_fold0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

IOU_T = 0.5
CLEANUP_MM3 = 1000.0


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_f(gt_lbl, pred_lbl):
    """ins_f1 = panoptic RQ. (전역 매칭; pelvic 라벨은 위치상 anatomy 분리됨)."""
    matching = match_fragments(gt_lbl, pred_lbl, method="hungarian")
    if not matching:
        return float("nan")
    tp = fn = 0
    matched_pred = set()
    for gid, pid in matching.items():
        if pid is None:
            fn += 1
            continue
        v = iou(pred_lbl == pid, gt_lbl == gid)
        if v >= IOU_T:
            tp += 1
            matched_pred.add(int(pid))
        else:
            fn += 1
    all_pred = {int(x) for x in set(np.unique(pred_lbl)) - {0}}
    fp = len(all_pred - matched_pred)
    return recognition_quality(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="predictions/d016_heldout_fold0")
    ap.add_argument("--gt_dir", default="predictions/gt_instance")
    ap.add_argument("--modes", nargs="+",
                    default=["legacy", "keep_all", "hybrid", "form_intact"])
    ap.add_argument("--min_vols", nargs="+", type=float,
                    default=[0, 250, 500, 1000])
    ap.add_argument("--cleanup", type=float, default=CLEANUP_MM3)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    cases = sorted(pred_dir.glob("PENGWIN_*.nii.gz"))
    print(f"[decode_sweep] {len(cases)} cases, cleanup={args.cleanup}mm3, IoU>={IOU_T}")

    # GT 미리 로드(+cleanup)
    gts = {}
    for p in cases:
        gi, ga = lps(gt_dir / p.name)
        sp = gi.GetSpacing(); spz = (sp[2], sp[1], sp[0])
        ga = np.where((ga >= 1) & (ga <= 150), ga, 0).astype(np.uint16)
        ga = remove_small_components(ga, spz, args.cleanup)
        gts[p.name] = (ga, spz)

    results = {}  # (mode, mv) -> {case: rq}
    for mode in args.modes:
        for mv in args.min_vols:
            row = {}
            for p in cases:
                sem_img, sem = lps(p)
                spx = sem_img.GetSpacing(); vv = float(np.prod(spx)); spz = (spx[2], spx[1], spx[0])
                inst = assemble_pelvic_instances_mode(
                    sem, vv, mode=mode, spacing_zyx=spz, min_volume_mm3=mv)
                inst = remove_small_components(inst.astype(np.uint16), spz, args.cleanup)
                ga, _ = gts[p.name]
                row[p.name] = rq_f(ga, inst)
            results[(mode, mv)] = row
            m = np.nanmean(list(row.values()))
            print(f"  mode={mode:11s} min_vol={mv:6.0f}  RQ_F_mean={m:.4f}", flush=True)

    # 최고 config + per-case
    best = max(results, key=lambda k: np.nanmean(list(results[k].values())))
    print(f"\n=== BEST: mode={best[0]} min_vol={best[1]} RQ_F_mean={np.nanmean(list(results[best].values())):.4f} ===")
    cids = [p.name.replace('.nii.gz', '').replace('PENGWIN_', '') for p in cases]
    print("case   " + "  ".join(f"{c:>5s}" for c in cids))
    for k in [("legacy", 500.0), best]:
        if k not in results: continue
        row = results[k]
        vals = [row[p.name] for p in cases]
        tag = f"{k[0][:8]},mv{int(k[1])}"
        print(f"{tag:14s} " + "  ".join(f"{v:5.2f}" for v in vals))
    print("DECODE_SWEEP_DONE")


if __name__ == "__main__":
    main()
