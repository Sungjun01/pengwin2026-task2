"""OOF(held-out) exhaustive decode 진단 — 병렬, 오염 없음.

목적: 재학습 0 으로 OOF ins_f1(=0.80 baseline) 을 올리는 decode 모드가 있나?
  legacy 가 병합하는 사크럼 fragment 를 다른 decode(keep_all/hybrid/form_intact)가
  더 잘 분리하면 → recall↑ → 003 의 놓친 조각도 회복 가능(재학습 없이, 안전).

병렬: case 별 워커가 모든 (mode × min_vol) 를 한 번에 decode+score (multiprocessing.Pool).
추가: legacy decode 의 missed-seam 회복가능성(HU 대조) → r_oof.

용법: python -m scripts.oof_decode_diagnosis
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from scipy.ndimage import generate_binary_structure, binary_dilation
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
GTD = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance"
OOF = "/tmp/oof_d016_sac"
IOU_T = 0.5
CLEANUP = 1000.0
MODES = ["legacy", "keep_all", "hybrid", "form_intact"]
MINVOLS = [250.0, 500.0, 1000.0]
LO, HI = 1, 50


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_f(gt_lbl, pred_lbl):
    matching = match_fragments(gt_lbl, pred_lbl, method="hungarian")
    if not matching:
        return float("nan")
    tp = fn = 0
    matched_pred = set()
    for gid, pid in matching.items():
        if pid is None:
            fn += 1; continue
        v = iou(pred_lbl == pid, gt_lbl == gid)
        if v >= IOU_T:
            tp += 1; matched_pred.add(int(pid))
        else:
            fn += 1
    all_pred = {int(x) for x in set(np.unique(pred_lbl)) - {0}}
    fp = len(all_pred - matched_pred)
    return recognition_quality(tp, fp, fn)


def sac_instances(lbl):
    return {int(v): (lbl == v) for v in np.unique(lbl) if LO <= v <= HI}


def missed_seam_contrast(inst_legacy, gt, ct):
    """legacy decode 가 병합한 GT 사크럼 fragment 쌍의 seam HU 대조 리스트."""
    gi = sac_instances(gt); pi = sac_instances(inst_legacy)
    out = []
    if len(gi) < 2:
        return out
    for pid, pmask in pi.items():
        covered = [gid for gid, gm in gi.items() if (pmask & gm).sum() > 0.2 * gm.sum()]
        for i in range(len(covered)):
            for j in range(i + 1, len(covered)):
                gA, gB = gi[covered[i]], gi[covered[j]]
                dA = binary_dilation(gA, ST, 1); dB = binary_dilation(gB, ST, 1)
                seam = (dA & gB) | (dB & gA)
                if seam.sum() < 3:
                    continue
                bone = ct[(gA | gB)]
                bone_med = float(np.median(bone[bone >= 0])) if (bone >= 0).any() else 200.0
                out.append(bone_med - float(np.median(ct[seam])))
    return out


def work(cid):
    pp = Path(OOF) / f"{cid}.nii.gz"; gp = Path(GTD) / f"{cid}.nii.gz"
    cnum = cid.replace("PENGWIN_", "")
    ip = Path(RAW) / cnum / "image.mha"
    if not (pp.exists() and gp.exists()):
        return cid, None, []
    sem_img, sem = lps(pp)
    spx = sem_img.GetSpacing(); vv = float(np.prod(spx)); spz = (spx[2], spx[1], spx[0])
    gi, ga = lps(gp)
    ga = np.where((ga >= 1) & (ga <= 150), ga, 0).astype(np.uint16)
    ga = remove_small_components(ga, spz, CLEANUP)
    per_cfg = {}
    inst_legacy = None
    for mode in MODES:
        for mv in MINVOLS:
            inst = assemble_pelvic_instances_mode(sem, vv, mode=mode, spacing_zyx=spz, min_volume_mm3=mv)
            inst = remove_small_components(inst.astype(np.uint16), spz, CLEANUP)
            per_cfg[(mode, mv)] = rq_f(ga, inst)
            if mode == "legacy" and mv == 500.0:
                inst_legacy = inst
    seams = []
    if ip.exists() and inst_legacy is not None:
        _, ct = lps(ip)
        seams = missed_seam_contrast(inst_legacy, ga, ct.astype(np.float32))
    return cid, per_cfg, seams


def main():
    frac = [l.strip() for l in open("/tmp/frac_sacrum_cases.txt") if l.strip()]
    cids = [f"PENGWIN_{c}" for c in frac]
    nproc = min(len(cids), max(1, (os.cpu_count() or 8) - 2))
    print(f"[oof_decode_diagnosis] {len(cids)} cases, {nproc} workers, "
          f"{len(MODES)}x{len(MINVOLS)} configs", flush=True)
    with Pool(nproc) as pool:
        res = pool.map(work, cids)

    # 집계: config -> per-case rq
    cfg_rows = {(m, v): {} for m in MODES for v in MINVOLS}
    all_seams = []
    for cid, per_cfg, seams in res:
        if per_cfg is None:
            continue
        for k, val in per_cfg.items():
            cfg_rows[k][cid] = val
        all_seams.extend(seams)

    print("\n=== OOF decode 스윕 (held-out, 깨끗) — config별 mean ins_f1 ===")
    ranked = sorted(cfg_rows.items(), key=lambda kv: -np.nanmean(list(kv[1].values())) if kv[1] else 0)
    for (m, v), row in ranked:
        if not row:
            continue
        mean = np.nanmean(list(row.values()))
        tag = "  <<baseline" if (m, v) == ("legacy", 500.0) else ""
        print(f"  mode={m:11s} min_vol={v:6.0f}  mean_ins_f1={mean:.4f}{tag}")
    base = np.nanmean(list(cfg_rows[("legacy", 500.0)].values()))
    bestk = ranked[0][0]; bestm = np.nanmean(list(cfg_rows[bestk].values()))
    print(f"\nbaseline legacy/mv500 = {base:.4f}")
    print(f"BEST = {bestk[0]}/mv{int(bestk[1])} = {bestm:.4f}  (Δ={bestm-base:+.4f})")
    print(f"판정: {'decode-only 개선 가능 (재학습 0)' if bestm-base > 0.005 else 'decode로는 개선 없음 → 재학습 필요'}")

    # per-case: legacy vs best
    print("\n=== per-case: legacy/mv500 vs BEST ===")
    print(f"{'case':6s} {'legacy':>7s} {'best':>7s} {'Δ':>6s}")
    for cid in cids:
        l = cfg_rows[("legacy", 500.0)].get(cid, float("nan"))
        b = cfg_rows[bestk].get(cid, float("nan"))
        d = b - l
        flag = "  <<개선" if d > 0.01 else ("  회귀" if d < -0.01 else "")
        print(f"{cid.replace('PENGWIN_',''):6s} {l:7.2f} {b:7.2f} {d:+6.2f}{flag}")

    # missed-seam 회복가능성
    arr = np.array(all_seams)
    print(f"\n=== OOF missed-seam 회복가능성 (legacy 병합 seam, n={len(arr)}) ===")
    if len(arr):
        n = len(arr)
        b1 = int((arr < 0).sum()); b2 = int(((arr >= 0) & (arr < 40)).sum())
        b3 = int(((arr >= 40) & (arr < 80)).sum()); b4 = int((arr >= 80).sum())
        print(f"  <0HU(뼈보다밝음): {b1} ({100*b1/n:.0f}%) | 0-40(영상한계): {b2} ({100*b2/n:.0f}%)")
        print(f"  40-80(회복): {b3} ({100*b3/n:.0f}%) | >=80(강신호): {b4} ({100*b4/n:.0f}%)")
        rec = b3 + b4
        print(f"  영상한계(<40)={100*(b1+b2)/n:.0f}%  회복가능(>=40) r_oof={100*rec/n:.0f}%")
        print(f"  판정: {'재학습 정당(r_oof>=25%)' if rec/n>=0.25 else '영상한계 지배'}")
    print("OOF_DIAGNOSIS_DONE")


if __name__ == "__main__":
    main()
