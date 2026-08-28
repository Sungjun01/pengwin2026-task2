"""면-완성(surface-completion) decode — 재학습 0. Step1=Case A(border 끊김) 의 처방.

원리: D016 의 사크럼 border(class2)가 seam 근처에 있으나 *끊겨* 있어 legacy watershed 가
틈으로 새어 두 조각을 병합. → border 를 morphological closing 으로 *이어* 연속 절단면을
만들고, core 에서 도려내 marker 를 분리한 뒤 distance-watershed.

  closed = binary_closing(border, k)         # 끊긴 면 잇기
  markers = CC( core & ~dilate(closed) )      # 절단면으로 core 분리 → 2+ marker
  inst = watershed(-EDT(fg), markers, mask=fg) # 면 따라 분할

closing k 스윕(0~3) × legacy 비교, 9 fold_3 사크럼-골절 케이스 RQ. k=0 ≈ legacy(틈 안 메움).

용법: python -m scripts.surface_completion_decode
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (label as cc_label, generate_binary_structure,
                           binary_closing, binary_dilation, distance_transform_edt)
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
SAC_CORE, SAC_BORD = 1, 2
CASES = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]
CLEANUP = 1000.0


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_sac(gt, pred, spz):
    g = np.where((gt >= 1) & (gt <= 50), gt, 0).astype(np.uint16)
    p = np.where((pred >= 1) & (pred <= 50), pred, 0).astype(np.uint16)
    g = remove_small_components(g, spz, CLEANUP); p = remove_small_components(p, spz, CLEANUP)
    m = match_fragments(g, p, method="hungarian")
    if not m:
        return float("nan")
    tp = fn = 0; mp = set()
    for gid, pid in m.items():
        if pid is None:
            fn += 1; continue
        if iou(p == pid, g == gid) >= 0.5:
            tp += 1; mp.add(int(pid))
        else:
            fn += 1
    fp = len({int(x) for x in np.unique(p)} - {0} - mp)
    return recognition_quality(tp, fn=fn, fp=fp)


def surface_completion_sacrum(sem, k):
    """사크럼만 면-완성 decode → 사크럼 instance 라벨(1..K). k=closing iters.
    속도: 사크럼 bbox 로 crop 후 처리(큰 볼륨 watershed ~10× 가속)."""
    full = np.zeros(sem.shape, dtype=np.uint16)
    fg_full = (sem == SAC_CORE) | (sem == SAC_BORD)
    if not fg_full.any():
        return full
    zz, yy, xx = np.where(fg_full)
    m = 3
    z0, z1 = max(0, zz.min() - m), min(sem.shape[0], zz.max() + m + 1)
    y0, y1 = max(0, yy.min() - m), min(sem.shape[1], yy.max() + m + 1)
    x0, x1 = max(0, xx.min() - m), min(sem.shape[2], xx.max() + m + 1)
    s = sem[z0:z1, y0:y1, x0:x1]
    core = (s == SAC_CORE); border = (s == SAC_BORD); fg = core | border
    closed = binary_closing(border, ST, iterations=k) if k > 0 else border
    cut = binary_dilation(closed, ST, iterations=1)
    seeds = core & ~cut
    lbl, n = cc_label(seeds, structure=ST)
    if n == 0:
        lbl, n = cc_label(core, structure=ST)
    for v in range(1, n + 1):
        if (lbl == v).sum() < 50:
            lbl[lbl == v] = 0
    if lbl.max() == 0:
        lbl, _ = cc_label(core, structure=ST)
    elev = -distance_transform_edt(fg)
    inst = watershed(elev, markers=lbl, mask=fg)
    full[z0:z1, y0:y1, x0:x1] = inst.astype(np.uint16)
    return full


def propagation_completion_sacrum(sem, ct, k=2, w_ct=1.0, w_dist=0.5, smooth=1.0):
    """(A) 균열-전파 완성: 끊긴 border를 *물리적 균열 경로*(어두운 저밀도 뼈)를 따라 완성.
    v24(형태학 closing=기하 최단)와 달리, watershed elevation에 CT 어두움을 ridge로 넣어
    경계가 *실제 균열선(어두움)*을 따라가게 함. marker는 closing(k)으로 seed, 경계는 CT-guided.
    smooth=CT 가우시안 평활(노이즈), w_ct=균열-유도 가중, w_dist=거리(thin-neck) fallback."""
    from scipy.ndimage import gaussian_filter
    full = np.zeros(sem.shape, dtype=np.uint16)
    fg_full = (sem == SAC_CORE) | (sem == SAC_BORD)
    if not fg_full.any():
        return full
    zz, yy, xx = np.where(fg_full); m = 3
    z0, z1 = max(0, zz.min() - m), min(sem.shape[0], zz.max() + m + 1)
    y0, y1 = max(0, yy.min() - m), min(sem.shape[1], yy.max() + m + 1)
    x0, x1 = max(0, xx.min() - m), min(sem.shape[2], xx.max() + m + 1)
    s = sem[z0:z1, y0:y1, x0:x1]; c = ct[z0:z1, y0:y1, x0:x1].astype(np.float32)
    core = (s == SAC_CORE); border = (s == SAC_BORD); fg = core | border
    closed = binary_closing(border, ST, iterations=k) if k > 0 else border
    seeds = core & ~binary_dilation(closed, ST, 1)
    lbl, n = cc_label(seeds, structure=ST)
    if n == 0:
        lbl, n = cc_label(core, structure=ST)
    for v in range(1, n + 1):
        if (lbl == v).sum() < 50:
            lbl[lbl == v] = 0
    if lbl.max() == 0:
        lbl, _ = cc_label(core, structure=ST)
    cs = gaussian_filter(c, smooth)
    fv = cs[fg]
    lo, hi = np.percentile(fv, 10), np.percentile(fv, 90)
    ctn = np.clip((cs - lo) / (hi - lo + 1e-6), 0, 1)   # 0=어두움(균열), 1=밝음(뼈)
    edt = distance_transform_edt(fg); edtn = edt / (edt.max() + 1e-6)
    elev = w_ct * (1.0 - ctn) - w_dist * edtn           # 어두움=high(ridge=균열), 깊은내부=low(basin)
    inst = watershed(elev, markers=lbl, mask=fg)
    full[z0:z1, y0:y1, x0:x1] = inst.astype(np.uint16)
    return full


def ct_evidence_guard(inst, ct, thr=60.0):
    """헛 split 제거: 인접 조각 경계에 골절 증거(어두운 seam) 없으면(<thr) 병합.
    진짜 골절=경계 어두움(대조 큼), 헛split=단단한 뼈 가름(대조~0). over-split(058) 교정.
    빠른 구현: 6-neighbor np.roll 1패스로 pair 경계 voxel 수집 + union-find 병합."""
    inst = inst.astype(np.int64).copy()
    ids = [int(v) for v in np.unique(inst) if v > 0]
    if len(ids) < 2:
        return inst.astype(np.uint16)
    bmed = {}
    for v in ids:
        b = ct[inst == v]; bmed[v] = float(np.median(b[b >= 0])) if (b >= 0).any() else 200.0
    M = max(ids) + 1
    from collections import defaultdict
    pair_c = defaultdict(list)
    for ax in range(3):
        for sh in (1, -1):
            rolled = np.roll(inst, sh, axis=ax)
            mask = (inst > 0) & (rolled > 0) & (inst != rolled)
            if not mask.any():
                continue
            aa = inst[mask]; bb = rolled[mask]; cc = ct[mask]
            lo = np.minimum(aa, bb); hi = np.maximum(aa, bb)
            pk = lo * M + hi
            for k in np.unique(pk):
                a = int(k // M); b = int(k % M)
                pair_c[(a, b)].append(cc[pk == k])
    parent = {v: v for v in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    for (a, b), arrs in pair_c.items():
        cvals = np.concatenate(arrs)
        contrast = max(bmed[a], bmed[b]) - cvals
        if len(contrast) < 3 or np.percentile(contrast, 90) < thr:  # 증거 약함 → 병합
            union(a, b)
    out = np.zeros_like(inst)
    for v in ids:
        out[inst == v] = find(v)
    return out.astype(np.uint16)


def surface_completion_guarded(sem, ct, k, thr):
    inst = surface_completion_sacrum(sem, k)
    return ct_evidence_guard(inst, ct, thr)


def main():
    print(f"=== 면-완성 decode 스윕 (9 fold_3 사크럼-골절, 사크럼 RQ) ===")
    rows = {}  # k -> {case: rq}
    legacy = {}
    for c in CASES:
        cid = f"PENGWIN_{c}"
        pp = Path(OOF) / f"{cid}.nii.gz"; gp = Path(RAW) / c / "label.mha"
        if not (pp.exists() and gp.exists()):
            continue
        pim, sem = lps(pp); sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(gp)
        leg = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        legacy[c] = rq_sac(gt, leg, spz)
        for k in [0, 1, 2, 3]:
            inst = surface_completion_sacrum(sem, k)
            rows.setdefault(k, {})[c] = rq_sac(gt, inst, spz)
    cs = list(legacy.keys())
    print(f"{'case':5s} {'legacy':>7s} " + " ".join(f"{'k='+str(k):>7s}" for k in [0, 1, 2, 3]))
    for c in cs:
        print(f"{c:5s} {legacy[c]:7.3f} " + " ".join(f"{rows[k][c]:7.3f}" for k in [0, 1, 2, 3]))
    legm = np.nanmean(list(legacy.values()))
    print(f"\n{'mean':5s} {legm:7.3f} " + " ".join(f"{np.nanmean(list(rows[k].values())):7.3f}" for k in [0, 1, 2, 3]))
    # best k + 회귀 체크
    print()
    for k in [1, 2, 3]:
        km = np.nanmean(list(rows[k].values()))
        reg = [c for c in cs if rows[k][c] < legacy[c] - 0.02]
        win = [c for c in cs if rows[k][c] > legacy[c] + 0.02]
        print(f"k={k}: mean Δ={km-legm:+.4f}  win={win}  regress={reg}")
    print("SURFCOMP_DONE")


if __name__ == "__main__":
    main()
