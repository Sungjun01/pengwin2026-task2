"""A: sliver-merge 임계 sweep on 사크럼골절 스코어보드 (001 over-seg fix 확인, 도박 0).

D016 예측(border-core 7-class)을 케이스당 1번 legacy decode(느린 border 확장) →
그 instance 에 sliver-merge 만 여러 임계로 적용(빠름) → pelvic+사크럼 RQ 채점.
목표: 더 느슨한 sliver-merge 가 (a) over-seg 케이스 RQ↑ 하면서 (b) 다른 케이스 regression 0.

공식정합: 1cm³ cleanup, DICOMOrient(LPS), IoU>=0.5.

용법: python -m scripts.sliver_merge_sweep --pred_dir /tmp/pred_d016_frac
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.pivot_assembly import AssemblyConfig, merge_sliver_fragments
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
CLEANUP = 1000.0
# v18 = 2000/5/1/30. 더 느슨 후보들 (max_sliver, ratio, dil, dist)
CONFIGS = {
    "v18(2000/5/1/30)": (2000, 5, 1, 30),
    "2500/4/2/40":      (2500, 4, 2, 40),
    "3000/3/3/60":      (3000, 3, 3, 60),
}


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_range(gt, pred, lo, hi, spz):
    """anatomy [lo,hi] 범위만 RQ (+cleanup)."""
    g = remove_small_components(np.where((gt >= lo) & (gt <= hi), gt, 0).astype(np.uint16), spz, CLEANUP)
    p = remove_small_components(np.where((pred >= lo) & (pred <= hi), pred, 0).astype(np.uint16), spz, CLEANUP)
    m = match_fragments(g, p, method="hungarian")
    if not m:
        return float("nan"), 0, 0
    tp = fn = 0; mp = set()
    for gid, pid in m.items():
        if pid is None: fn += 1; continue
        if iou(p == pid, g == gid) >= 0.5: tp += 1; mp.add(int(pid))
        else: fn += 1
    fp = len({int(x) for x in set(np.unique(p)) - {0}} - mp)
    ng = len([v for v in np.unique(g) if v > 0]); npr = len([v for v in np.unique(p) if v > 0])
    return recognition_quality(tp, fp, fn), ng, npr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/tmp/pred_d016_frac")
    args = ap.parse_args()
    preds = sorted(Path(args.pred_dir).glob("PENGWIN_*.nii.gz"))
    print(f"[sliver_sweep] {len(preds)} cases")

    # 케이스당 1회 legacy decode (sliver-merge 전), GT 로드
    base = {}
    for p in preds:
        cid = p.name.replace("PENGWIN_", "").replace(".nii.gz", "")
        pim, sem_full = lps(p)
        sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt_full = lps(Path(RAW) / cid / "label.mha")
        # pelvic foreground bbox 로 crop (border-expansion 속도 수십배)
        fg = (sem_full > 0) | ((gt_full >= 1) & (gt_full <= 150))
        if not fg.any():
            continue
        zz, yy, xx = np.where(fg); pad = 6
        slz = (slice(max(0, zz.min()-pad), zz.max()+pad+1),
               slice(max(0, yy.min()-pad), yy.max()+pad+1),
               slice(max(0, xx.min()-pad), xx.max()+pad+1))
        sem = sem_full[slz]; gt = gt_full[slz]
        inst0 = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        base[cid] = (inst0, gt, vv, spz)
        print(f"  decoded {cid}", flush=True)

    # 각 config: sliver-merge 적용 → 사크럼+pelvic RQ
    print("\n" + "=" * 70)
    rows = {}
    for name, (msl, ratio, dil, dist) in CONFIGS.items():
        cfg = AssemblyConfig(max_sliver_volume_mm3=msl, min_volume_ratio=ratio,
                             dilation_iters=dil, max_centroid_distance_mm=dist)
        sac_rq, pel_rq = [], []
        per = {}
        for cid, (inst0, gt, vv, spz) in base.items():
            inst = merge_sliver_fragments(inst0, vv, cfg, spz)
            s, sg, sp_ = rq_range(gt, inst, 1, 50, spz)
            pl, _, _ = rq_range(gt, inst, 1, 150, spz)
            sac_rq.append(s); pel_rq.append(pl)
            per[cid] = (s, sg, sp_, pl)
        rows[name] = per
        print(f"{name:18s}  사크럼RQ={np.nanmean(sac_rq):.3f}  pelvicRQ={np.nanmean(pel_rq):.3f}")

    # v18 대비 per-case 변화 (regression 탐지)
    print("\n=== v18 대비 사크럼 RQ 변화 (over-seg fix vs regression) ===")
    base_name = "v18(2000/5/1/30)"
    print(f"{'case':6s} {'nGTsac':>6s} {'nPredSac':>8s} {'v18':>5s} " + " ".join(f"{n.split('/')[0]:>6s}" for n in CONFIGS if n != base_name))
    for cid in sorted(base):
        v18s = rows[base_name][cid][0]
        sg = rows[base_name][cid][1]; sp_ = rows[base_name][cid][2]
        deltas = []
        for n in CONFIGS:
            if n == base_name: continue
            deltas.append(rows[n][cid][0] - v18s)
        flag = ""
        if any(d > 0.02 for d in deltas): flag = " ↑개선"
        if any(d < -0.02 for d in deltas): flag += " ↓회귀"
        print(f"{cid:6s} {sg:>6d} {sp_:>8d} {v18s:5.2f} " + " ".join(f"{d:+6.2f}" for d in deltas) + flag)
    print("SLIVER_SWEEP_DONE")


if __name__ == "__main__":
    main()
