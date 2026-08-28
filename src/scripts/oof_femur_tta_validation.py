"""TTA held-out 검증 — D004 fold_0(held-out)로 no-TTA vs TTA 경계지표 비교.

질문: TTA(mirror) + overlap(tile_step 0.5) + gaussian이 *안 본* femur의 표면(HD95/ASSD/dice)을
정말 개선하나? (memorized 아님 — fold_0 val 34케이스, 모델이 학습 안 함.)

각 케이스: D004 fold_0 → femur binary (A: no-TTA / B: TTA) → GT femur binary와 표면거리 비교.
TTA가 HD95/ASSD를 *낮추면* 배포 가치, 중립이면 femur TTA는 레버 아님.

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 -m scripts.oof_femur_tta_validation
"""
from __future__ import annotations
import sys, os, json, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scripts.inference_v7_d015 import predict_softmax

ST = generate_binary_structure(3, 3)
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset004_PENGWIN_FemurAnatomy/splits_final.json")
FOLD0 = "/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"


def init_pred(use_mirroring, tile_step, gaussian):
    p = nnUNetPredictor(tile_step_size=tile_step, use_gaussian=gaussian, use_mirroring=use_mirroring,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(FOLD0, use_folds=("0",), checkpoint_name="checkpoint_best.pth")
    return p


def surf(m):
    return m & ~binary_erosion(m, ST)


def surface_metrics(pred, gt, spz):
    if not pred.any() or not gt.any():
        return float("nan"), float("nan"), float("nan")
    ps, gs = surf(pred), surf(gt)
    d_pg = distance_transform_edt(~gs, sampling=spz)[ps]
    d_gp = distance_transform_edt(~ps, sampling=spz)[gs]
    alld = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(alld, 95)); assd = float(alld.mean())
    dice = 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum())
    return hd95, assd, float(dice)


def main():
    cids = [v.split("_")[-1] for v in json.load(open(SPLITS))[0]["val"]]
    pA = init_pred(False, 1.0, False)        # no-TTA
    pB = init_pred(True, 0.5, True)          # TTA + overlap + gaussian
    tmp = Path(tempfile.mkdtemp(prefix="ttaval_"))
    aA = {k: [] for k in ["hd95", "assd", "dice"]}; aB = {k: [] for k in aA}
    print(f"[TTA val] D004 fold_0 (held-out)  n={len(cids)}\n", flush=True)
    print(f"{'case':6s} {'HD95_noTTA':>10s} {'HD95_TTA':>9s} {'ΔHD95':>7s} {'ASSD_no':>8s} {'ASSD_TTA':>9s}", flush=True)
    for cid in cids:
        ip = RAW / cid / "image.mha"
        if not ip.exists():
            continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(ip)), "LPS")
        gt = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(RAW/cid/"label.mha")), "LPS"))
        femgt = (gt >= 151) & (gt <= 200)
        sx, sy, sz = ct.GetSpacing(); spz = (sz, sy, sx)
        cp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct, str(cp))
        binA = (predict_softmax(pA, cp, tmp/"a")[1] > 0.5)
        binB = (predict_softmax(pB, cp, tmp/"b")[1] > 0.5)
        hA, sA, dA = surface_metrics(binA, femgt, spz)
        hB, sB, dB = surface_metrics(binB, femgt, spz)
        for k, v in zip(["hd95", "assd", "dice"], [hA, sA, dA]): aA[k].append(v)
        for k, v in zip(["hd95", "assd", "dice"], [hB, sB, dB]): aB[k].append(v)
        print(f"{cid:6s} {hA:10.2f} {hB:9.2f} {hB-hA:+7.2f} {sA:8.3f} {sB:9.3f}", flush=True)

    print(f"\n=== held-out 평균 (n={len(aA['hd95'])}) — TTA가 경계를 낮추나? ===")
    for k in ["hd95", "assd", "dice"]:
        mA, mB = np.nanmean(aA[k]), np.nanmean(aB[k])
        med_A, med_B = np.nanmedian(aA[k]), np.nanmedian(aB[k])
        print(f"  {k:5s}: no-TTA mean={mA:7.3f}(med {med_A:.3f})  TTA mean={mB:7.3f}(med {med_B:.3f})  Δ={mB-mA:+.3f}")
    print("TTA_VAL_DONE", flush=True)


if __name__ == "__main__":
    main()
