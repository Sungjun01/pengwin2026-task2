"""사크럼 후처리 decode 방법 비교 (D016 출력 재처리, 모델 재학습 0).

질문: D016 이 사크럼 골절선을 봤는데 legacy decode 가 못 쓴 건가?
→ 같은 D016 출력(argmax 7-class + soft 확률)을 여러 decode 로 재처리해
   GT 사크럼 조각 수에 얼마나 근접하나 비교.

방법:
  A. legacy        : core(==1) CC marker (v18 기본)
  B. eroded-core   : core 를 erode 후 CC (얇은 목 끊어 더 분리)
  C. soft-border-ws: core CC marker + border 확률을 watershed energy (모델 seam 사용)
  D. dist-watershed: 사크럼 foreground distance-transform peak marker (touching 분리)

채점: GT 사크럼(1-50) 대비 IoU>=0.5 매칭 + 1cm³ cleanup.

용법: python -m scripts.sacrum_decode_experiment --case 097 \
        --pred /tmp/pred_d016_097/PENGWIN_097.nii.gz --probs /tmp/pred_d016_097/PENGWIN_097.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    label as cc_label, generate_binary_structure, binary_erosion,
    distance_transform_edt, maximum_filter,
)
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
GTD = "predictions/gt_instance"
SAC_CORE, SAC_BORD = 1, 2  # D016 7-class: 1=sac_core, 2=sac_border
LO, HI = 1, 50
MIN_VOL = 500.0
CLEANUP = 1000.0


def lps_img(p):
    return sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")


def assign(markers_lab, energy, mask, vv, spz):
    """watershed + min_vol + cleanup → 사크럼 id 부여, 조각 수 반환."""
    lab = watershed(energy, markers_lab, mask=mask)
    out = np.zeros(mask.shape, np.uint16)
    ids = sorted(((int((lab == c).sum()), c) for c in range(1, int(lab.max()) + 1)), reverse=True)
    nid = LO
    for v, c in ids:
        if v * vv < MIN_VOL or nid > HI:
            continue
        out[lab == c] = nid; nid += 1
    out = remove_small_components(out, spz, CLEANUP)
    return out


def score_sacrum(inst, gt, spz):
    gts = remove_small_components(np.where((gt >= LO) & (gt <= HI), gt, 0).astype(np.uint16), spz, CLEANUP)
    matching = match_fragments(gts, inst, method="hungarian")
    tp = fn = 0; mp = set()
    for gid, pid in matching.items():
        if pid is None: fn += 1; continue
        if iou(inst == pid, gts == gid) >= 0.5: tp += 1; mp.add(int(pid))
        else: fn += 1
    fp = len({int(x) for x in set(np.unique(inst)) - {0}} - mp)
    rq = recognition_quality(tp, fp, fn)
    ngt = len([v for v in np.unique(gts) if v > 0])
    npr = len([v for v in np.unique(inst) if v > 0])
    return rq, ngt, npr, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--probs", required=True)
    args = ap.parse_args()

    pimg = lps_img(args.pred)
    sem_full = sitk.GetArrayFromImage(pimg)
    sp = pimg.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
    gt_full = sitk.GetArrayFromImage(lps_img(Path(GTD) / f"PENGWIN_{args.case}.nii.gz")).astype(int)

    # 사크럼 영역(core|border)만 crop → 속도 수십배 (풀 볼륨 watershed 는 타임아웃)
    fg_full = (sem_full == SAC_CORE) | (sem_full == SAC_BORD)
    if not fg_full.any():
        print("사크럼 없음"); return
    zz, yy, xx = np.where(fg_full | ((gt_full >= LO) & (gt_full <= HI)))
    pad = 8
    z0, z1 = max(0, zz.min() - pad), min(sem_full.shape[0], zz.max() + pad + 1)
    y0, y1 = max(0, yy.min() - pad), min(sem_full.shape[1], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(sem_full.shape[2], xx.max() + pad + 1)
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    sem = sem_full[sl]; gt = gt_full[sl]
    print(f"crop {sem_full.shape} -> {sem.shape}", flush=True)

    core = (sem == SAC_CORE)
    bord = (sem == SAC_BORD)
    fg = core | bord
    if not core.any():
        print("사크럼 core 없음"); return

    print(f"=== case {args.case} 사크럼 decode 비교 (GT 대비) ===")
    print(f"{'method':16s} {'RQ':>5s} {'nGT':>4s} {'nPred':>5s} {'TP':>3s} {'FP':>3s} {'FN':>3s}")

    results = {}
    # A. legacy: core CC marker, energy = -dist(fg)
    mk, n = cc_label(core, structure=ST)
    distfg = distance_transform_edt(fg, sampling=spz)
    results["A_legacy"] = assign(mk, -distfg, fg, vv, spz)

    # B. eroded-core (얇은 목 끊기): erode core 후 CC
    for it in (1, 2):
        ec = binary_erosion(core, structure=ST, iterations=it)
        mk2, n2 = cc_label(ec, structure=ST)
        if n2 == 0: mk2, n2 = cc_label(core, structure=ST)
        results[f"B_erode{it}"] = assign(mk2, -distfg, fg, vv, spz)

    # C. border-distance watershed: border 를 절단면으로 (border 에서 먼 곳이 core 중심)
    dist_from_border = distance_transform_edt(~bord, sampling=spz)
    core_seed = (dist_from_border > 1.5) & core  # border 에서 1.5mm 이상 = 확실한 core
    mk3, n3 = cc_label(core_seed, structure=ST)
    if n3 == 0: mk3, n3 = cc_label(core, structure=ST)
    results["C_borderws"] = assign(mk3, -dist_from_border, fg, vv, spz)

    # D. dist-watershed peak markers (touching 분리)
    peaks = (maximum_filter(distfg, size=5) == distfg) & (distfg > 2.0)
    mk4, n4 = cc_label(peaks, structure=ST)
    if n4 == 0: mk4, n4 = cc_label(core, structure=ST)
    results["D_distws"] = assign(mk4, -distfg, fg, vv, spz)

    for name, inst in results.items():
        rq, ngt, npr, tp, fp, fn = score_sacrum(inst, gt, spz)
        print(f"{name:16s} {rq:5.2f} {ngt:4d} {npr:5d} {tp:3d} {fp:3d} {fn:3d}")
    print("SACRUM_DECODE_DONE")


if __name__ == "__main__":
    main()
