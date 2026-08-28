"""femur instance method 로컬 스코어보드 (GT 기반).

femur 학습 케이스(251+, GT = raw_data/PENGWIN_train/<cid>/label.mha, fragments 151+)에
대해 D004 binary 의 instance 분리 방법 3종(watershed / lr_split / cc) + D013 border-core
를 official metric 으로 비교. LPS 정합 후 채점.

용법: CUDA_VISIBLE_DEVICES=0 python -m scripts.femur_method_scoreboard
"""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.ndimage import label as cc_label, generate_binary_structure
from scripts.inference_v7_d015 import (
    init_predictor, predict_softmax,
    per_anatomy_lr_split_instance, per_anatomy_watershed_instance,
)
from scripts.pivot_form_postproc import border_core_to_instances
from scripts.femur_ct_watershed import per_anatomy_ct_watershed_instance
from evaluation.pengwin_eval import evaluate_case

ST26 = generate_binary_structure(3, 3)
FEMUR = (151, 200)
MIN_VOL = 500.0
RES = Path("/home/schoaq/taehee/nnUNet_data/results")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
D004_DIR = RES / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
D013_DIR = RES / "Dataset013_PENGWIN_BorderCoreFemur" / "nnUNetTrainerBorderWeighted__nnUNetPlans__3d_fullres"
import os as _os
_def_cases = "251,252,253,254,255,256,259,260,263,266,277,278"
CASES = [c.strip() for c in _os.environ.get("PENGWIN_FEMUR_CASES", _def_cases).split(",") if c.strip()]


def cc_instances(binary, voxel_vol, lo=151, hi=200, min_vol=MIN_VOL):
    out = np.zeros(binary.shape, np.uint16)
    labels, n = cc_label(binary > 0, structure=ST26)
    sizes = sorted(((int((labels == c).sum()), c) for c in range(1, n + 1)), reverse=True)
    nid = lo
    for vox, c in sizes:
        if vox * voxel_vol < min_vol or nid > hi:
            continue
        out[labels == c] = nid; nid += 1
    return out


def write_lps(arr, ref_lps_img):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref_lps_img)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def main():
    tmp = Path(tempfile.mkdtemp(prefix="femsb_"))
    print(f"[scoreboard] D004 predictor ...", flush=True)
    pred004 = init_predictor(str(D004_DIR), tile_step_size=1.0)
    print(f"[scoreboard] D013 predictor ...", flush=True)
    pred013 = init_predictor(str(D013_DIR), tile_step_size=1.0)

    methods = ["watershed", "ct_ws", "lr_split", "cc", "d013"]
    agg = {m: [] for m in methods}
    print(f"{'case':6s} {'gtN':>3s} " + " ".join(f"{m+'_f1':>11s}" for m in methods), flush=True)

    for cid in CASES:
        ct_img = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "image.mha")), "LPS")
        gt_img = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        gt_arr = sitk.GetArrayFromImage(gt_img).astype(int)
        gtN = len([v for v in np.unique(gt_arr) if 151 <= v <= 200])
        sx, sy, sz = ct_img.GetSpacing(); vv = float(sx * sy * sz); spz = (sz, sy, sx)
        gt_f = write_lps(np.where((gt_arr >= 151) & (gt_arr <= 200), gt_arr, 0), gt_img)
        ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct_img, str(ct_path))

        # D004 binary
        probs004 = predict_softmax(pred004, ct_path, tmp / "d004")
        fem_bin = (probs004[1].astype(np.float32) > 0.5).astype(np.uint8)
        # D013 semantic (argmax: 0 bg / 1 core / 2 border)
        probs013 = predict_softmax(pred013, ct_path, tmp / "d013")
        sem013 = probs013.argmax(axis=0).astype(np.uint8)

        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        insts = {
            "watershed": per_anatomy_watershed_instance(fem_bin, 1, FEMUR, vv, MIN_VOL),
            "ct_ws": per_anatomy_ct_watershed_instance(fem_bin, ct_arr, 1, FEMUR, vv, MIN_VOL,
                                                       w_ct=1.0, w_dist=0.5),
            "lr_split": per_anatomy_lr_split_instance(fem_bin, 1, FEMUR, vv, MIN_VOL),
            "cc": cc_instances(fem_bin, vv),
            "d013": border_core_to_instances(sem013, 1, 2, FEMUR, vv, min_volume_mm3=MIN_VOL,
                                             form_cfg=None, spacing_zyx=spz),
        }
        row = {}
        for m, inst in insts.items():
            pf = write_lps(inst, ct_img)
            r = evaluate_case(pf, gt_f)
            agg[m].append(r); row[m] = r
            os.unlink(pf)
        os.unlink(gt_f)
        npred = {m: len([v for v in np.unique(insts[m]) if 151 <= v <= 200]) for m in methods}
        print(f"{cid:6s} {gtN:>3d} " + " ".join(
            f"{row[m]['RQ_F']:.2f}({npred[m]})".rjust(11) for m in methods), flush=True)

    print("\n=== 평균 (RQ_F=ins_f1, LDSC_F, HD95_F) ===", flush=True)
    for m in methods:
        rq = np.nanmean([r["RQ_F"] for r in agg[m]])
        ld = np.nanmean([r["LDSC_F"] for r in agg[m]])
        hd = np.nanmean([r["HD95_F_mm"] for r in agg[m]])
        print(f"  {m:10s} RQ_F={rq:.3f}  LDSC_F={ld:.3f}  HD95={hd:.2f}", flush=True)
    print("FEMUR_SB_DONE", flush=True)


if __name__ == "__main__":
    main()
