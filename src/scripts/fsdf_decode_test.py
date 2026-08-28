"""FS-DF 예측 decode + 채점 (신방법론 실전 테스트).

D020(pelvic, 12-class anatomy×4bin) 예측 → 거리장 복원 → watershed → instance → 채점.
검증된 fsdf_experiment.fsdf_decode 방식(core=거리>3mm CC marker)을 예측 bin 에 맞게:
  - bin → 대표거리 복원: bin1=0.75, bin2=2.25, bin3=4.5, bin4=8.0 (mm, THRESH 중앙값)
  - soft: gaussian 약간 흐림(연속성 = watershed 강건성, GT천장 실험서 확인)
  - core marker = bin>=3 (>3mm), watershed(-field, markers, mask=bone)

비교: legacy border-core decode (D016 0.853 / D019 0.879, 공정 in-sample) vs FS-DF.
핵심 관전: 017 (D016 0.67 / D019 0.73, 둘 다 못 푼 stuck) 가 FS-DF 로 오르나.

용법: python -m scripts.fsdf_decode_test --pred_dir <12class예측> --gt_dir predictions/gt_instance
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, label as cc_label,
    generate_binary_structure,
)
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

ST = generate_binary_structure(3, 3)
IOU_T = 0.5
CLEANUP_MM3 = 1000.0
MIN_VOL = 500.0
# anatomy: (lo, hi, class_base)  — D020 스키마
ANAT = {"sacrum": (1, 50, 0), "leftHip": (51, 100, 4), "rightHip": (101, 150, 8)}
# bin → 대표거리(mm). THRESH=[1.5,3,6] 의 구간 중앙값. bin4 는 open → 8.0.
BIN_DIST = {1: 0.75, 2: 2.25, 3: 4.5, 4: 8.0}
CORE_BINS = (3, 4)  # >3mm = marker (fsdf_experiment core_t_mm=3.0 과 정합)


def fsdf_decode_pelvic(sem12, spacing_xyz, sigma=1.0):
    """12-class FS-DF 예측 → pelvic instance(1-150)."""
    out = np.zeros(sem12.shape, np.uint16)
    vv = float(np.prod(spacing_xyz))
    for nm, (lo, hi, base) in ANAT.items():
        # 이 anatomy 의 4 bin 합 = bone_mask
        bone = np.isin(sem12, [base + 1, base + 2, base + 3, base + 4])
        if not bone.any():
            continue
        # 거리장 복원 (bin → 대표거리)
        field = np.zeros(sem12.shape, np.float32)
        for b in (1, 2, 3, 4):
            field[sem12 == base + b] = BIN_DIST[b]
        if sigma > 0:
            field = gaussian_filter(field, sigma)
            field[~bone] = 0.0
        # core marker = bin>=3
        core = np.isin(sem12, [base + c for c in CORE_BINS]) & bone
        markers, n = cc_label(core, structure=ST)
        if n == 0:
            markers, n = cc_label(bone, structure=ST)
        lab = watershed(-field, markers, mask=bone)
        # min_vol filter + id 할당 (anatomy 범위 내)
        ids = sorted(
            ((int((lab == c).sum()), c) for c in range(1, int(lab.max()) + 1)),
            reverse=True,
        )
        nid = lo
        for v, c in ids:
            if v * vv < MIN_VOL:
                continue
            if nid > hi:
                break
            out[lab == c] = nid
            nid += 1
    return out


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_f(gt, pred):
    matching = match_fragments(gt, pred, method="hungarian")
    if not matching:
        return float("nan")
    tp = fn = 0
    mp = set()
    for gid, pid in matching.items():
        if pid is None:
            fn += 1
            continue
        if iou(pred == pid, gt == gid) >= IOU_T:
            tp += 1
            mp.add(int(pid))
        else:
            fn += 1
    fp = len({int(x) for x in set(np.unique(pred)) - {0}} - mp)
    return recognition_quality(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", default="predictions/gt_instance")
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()
    preds = sorted(Path(args.pred_dir).glob("PENGWIN_*.nii.gz"))
    print(f"[fsdf_decode_test] {len(preds)} cases, sigma={args.sigma}, cleanup={CLEANUP_MM3}")
    print("case   ngt  npred  RQ_F")
    rqs = []
    for p in preds:
        pim, sem = lps(p)
        sp = pim.GetSpacing(); spz = (sp[2], sp[1], sp[0])
        inst = fsdf_decode_pelvic(sem, sp)
        inst = remove_small_components(inst.astype(np.uint16), spz, CLEANUP_MM3)
        gi, ga = lps(Path(args.gt_dir) / p.name)
        ga = np.where((ga >= 1) & (ga <= 150), ga, 0).astype(np.uint16)
        ga = remove_small_components(ga, spz, CLEANUP_MM3)
        ngt = len([v for v in np.unique(ga) if v > 0])
        npr = len([v for v in np.unique(inst) if v > 0])
        rq = rq_f(ga, inst)
        rqs.append(rq)
        cid = p.name.replace("PENGWIN_", "").replace(".nii.gz", "")
        print(f"{cid:5s}  {ngt:3d}  {npr:4d}  {rq:.3f}", flush=True)
    print(f">>> FS-DF RQ_F mean = {np.nanmean(rqs):.4f}")
    print("(비교: D016 0.853 / D019 0.879 — 둘 다 예측채널·in-sample)")
    print("FSDF_DECODE_TEST_DONE")


if __name__ == "__main__":
    main()
