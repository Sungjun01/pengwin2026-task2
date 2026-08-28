"""FS-DF (Fracture-Surface Distance Field) 실험 1 — 강건성 검증 (GT 기반, 학습 없음).

가설: 골절면까지의 *연속 distance field* 를 watershed energy 로 쓰면, 예측이 살짝
틀려도(노이즈) 붙은 조각이 분리된다 — thin border 보다 강건. 이게 우리 001-type
merge 를 푸는 핵심.

방법:
  1. GT instance label → 뼈별 골절면(인접 fragment 경계) 추출 → EDT → FS-DF (interior 큼, seam 0).
  2. FS-DF 를 gaussian σ 로 흐림 (예측 오차 시뮬레이션).
  3. FS-DF 로 marker-watershed decode → instance → GT 채점.
  4. σ 별 ins_f1 곡선 vs 현 decode(border-core, femur 0.58 / pelvic 0.76) vs 천장 0.96.

뼈별로 처리하고 femur(151-200) + pelvic(1-150) 둘 다.
"""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as cc_label, generate_binary_structure, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skimage.segmentation import watershed
from evaluation.pengwin_eval import evaluate_case

ST = generate_binary_structure(3, 3)
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
BONES = {"sacrum": (1, 50), "leftHip": (51, 100), "rightHip": (101, 150), "femur": (151, 200)}
FEMUR_CASES = ["251", "252", "256"]   # 4/3/9 fragments
PELVIC_CASES = ["001", "017"]         # pelvic
SIGMAS = [0.0, 2.0, 4.0]
MIN_VOL = 500.0


def fracture_surface(inst, bone_mask):
    """뼈 안에서 서로 다른 fragment 가 맞닿는 voxel = 골절면."""
    surf = np.zeros(inst.shape, bool)
    for dz, dy, dx in [(1,0,0),(0,1,0),(0,0,1)]:
        a = inst[1:,:,:] if dz else (inst[:,1:,:] if dy else inst[:,:,1:])
        b = inst[:-1,:,:] if dz else (inst[:,:-1,:] if dy else inst[:,:,:-1])
        diff = (a != b) & (a > 0) & (b > 0)
        if dz:
            surf[1:,:,:] |= diff; surf[:-1,:,:] |= diff
        elif dy:
            surf[:,1:,:] |= diff; surf[:,:-1,:] |= diff
        else:
            surf[:,:,1:] |= diff; surf[:,:,:-1] |= diff
    return surf & bone_mask


def fsdf_decode(bone_mask, fsdf, vv, lo, hi, core_t_mm=3.0):
    """border-core식: FS-DF > t(mm) 로 seam 주변을 깎아 core CC = marker.
    intact 뼈(seam 없음) → core 1개 → 1 fragment. 골절 → seam에서 갈라진 core들."""
    out = np.zeros(bone_mask.shape, np.uint16)
    if not bone_mask.any():
        return out
    core = (fsdf > core_t_mm) & bone_mask
    markers, n = cc_label(core, structure=ST)
    if n == 0:  # 다 깎였으면 통째로 1개
        markers, n = cc_label(bone_mask, structure=ST)
    labels = watershed(-fsdf, markers, mask=bone_mask)
    # min_vol filter + id 할당
    ids = [(int((labels == c).sum()), c) for c in range(1, labels.max() + 1)]
    ids = sorted([(v, c) for v, c in ids if v * vv >= MIN_VOL], reverse=True)
    nid = lo
    for v, c in ids:
        if nid > hi: break
        out[labels == c] = nid; nid += 1
    return out


def lps(p):
    img = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return img, sitk.GetArrayFromImage(img).astype(int)


def score(pred, gt, ref):
    pf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    gf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    pi = sitk.GetImageFromArray(pred.astype(np.uint16)); pi.CopyInformation(ref); sitk.WriteImage(pi, pf)
    gi = sitk.GetImageFromArray(gt.astype(np.uint16)); gi.CopyInformation(ref); sitk.WriteImage(gi, gf)
    r = evaluate_case(pf, gf); os.unlink(pf); os.unlink(gf); return r


def run(cases, anat_lo, anat_hi, tag):
    print(f"\n=== {tag}  (현 decode {'femur 0.58' if anat_lo>150 else 'pelvic 0.76'} / 천장 0.96) ===", flush=True)
    print("case  gtN  " + "  ".join(f"σ={s}" for s in SIGMAS), flush=True)
    agg = {s: [] for s in SIGMAS}
    for cid in cases:
        img, inst = lps(RAW / cid / "label.mha")
        inst = np.where((inst >= anat_lo) & (inst <= anat_hi), inst, 0)
        gtN = len([v for v in np.unique(inst) if v != 0])
        sp = img.GetSpacing(); vv = float(np.prod(sp))
        # FS-DF target (bone별로 합산)
        fsdf = np.zeros(inst.shape, np.float32)
        full_bone = np.zeros(inst.shape, bool)
        for nm, (lo, hi) in BONES.items():
            if lo < anat_lo or hi > anat_hi: continue
            bm = (inst >= lo) & (inst <= hi)
            if not bm.any(): continue
            full_bone |= bm
            surf = fracture_surface(inst, bm)
            d = distance_transform_edt(~surf, sampling=sp[::-1]).astype(np.float32)
            fsdf[bm] = d[bm]
        row = []
        for s in SIGMAS:
            f = gaussian_filter(fsdf, s) if s > 0 else fsdf
            pred = np.zeros(inst.shape, np.uint16)
            for nm, (lo, hi) in BONES.items():
                if lo < anat_lo or hi > anat_hi: continue
                bm = (inst >= lo) & (inst <= hi)
                if bm.any():
                    pred += fsdf_decode(bm, f, vv, lo, hi)
            r = score(pred, inst, img)
            agg[s].append(r["RQ_F"]); row.append(r["RQ_F"])
        print(f"{cid:5s} {gtN:>3d}  " + "  ".join(f"{x:.2f}" for x in row), flush=True)
    print("평균: " + "  ".join(f"σ={s}:{np.nanmean(agg[s]):.3f}" for s in SIGMAS), flush=True)


def main():
    run(FEMUR_CASES, 151, 200, "FEMUR FS-DF 강건성")
    run(PELVIC_CASES, 1, 150, "PELVIC FS-DF 강건성")
    print("FSDF_EXP_DONE", flush=True)


if __name__ == "__main__":
    main()
