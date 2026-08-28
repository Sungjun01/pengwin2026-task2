"""femur fold_0 held-out OOF — 8지표 전부로 채점 (pseudo Dice 아님, 진짜 평가).

splits_final.json fold_0 val(34 held-out femur 케이스, 학습 안 봄)에 대해
주어진 checkpoint로 추론 → watershed decode → evaluate_case 8지표.
baseline vs CF-Surf 를 *같은 decode*로 비교 → CF-Surf의 경계 이득(HD95/ASSD) 검증.

지표: anatomy 3 (IoU_A/HD95_A/ASSD_A) + fracture 5 (IoU_F/HD95_F/ASSD_F/RQ_F=ins_f1/LDSC_F=local_dice).

env:
  PENGWIN_OOF_CKPT   checkpoint dir (기본 baseline fold_0)
  PENGWIN_OOF_FOLD   use_folds (기본 "0")
용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 -m scripts.oof_femur_8metric
"""
from __future__ import annotations
import sys, os, json, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scripts.inference_v7_d015 import predict_softmax, per_anatomy_watershed_instance
from evaluation.pengwin_eval import evaluate_case

FEMUR = (151, 200)
MIN_VOL = 500.0
RES = Path("/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset004_PENGWIN_FemurAnatomy/splits_final.json")
CKPT_DIR = os.environ.get("PENGWIN_OOF_CKPT",
    str(RES / "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"))
FOLD = os.environ.get("PENGWIN_OOF_FOLD", "0")


def init_fold_predictor(model_dir, fold):
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True,
                        device=torch.device("cuda", 0), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=(fold,),
                                           checkpoint_name="checkpoint_best.pth")
    return p


def write_lps(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def main():
    val = json.load(open(SPLITS))[0]["val"]
    cids = [v.split("_")[-1] for v in val]
    print(f"[OOF] ckpt={Path(CKPT_DIR).name} fold={FOLD}  held-out n={len(cids)}", flush=True)
    pred = init_fold_predictor(CKPT_DIR, FOLD)
    tmp = Path(tempfile.mkdtemp(prefix="oof8_"))
    keys = ["IoU_A", "HD95_A_mm", "ASSD_A_mm", "IoU_F", "HD95_F_mm", "ASSD_F_mm", "RQ_F", "LDSC_F"]
    agg = {k: [] for k in keys}
    print(f"{'case':10s} " + " ".join(f"{k:>9s}" for k in keys), flush=True)
    for cid in cids:
        img_p = RAW / cid / "image.mha"
        if not img_p.exists():
            print(f"{cid}: image 없음 skip", flush=True); continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(img_p)), "LPS")
        gt = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        sx, sy, sz = ct.GetSpacing(); vv = float(sx * sy * sz)
        ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct, str(ct_path))
        probs = predict_softmax(pred, ct_path, tmp / "out")
        fem_bin = (probs[1].astype(np.float32) > 0.5).astype(np.uint8)
        inst = per_anatomy_watershed_instance(fem_bin, 1, FEMUR, vv, MIN_VOL)
        gt_arr = sitk.GetArrayFromImage(gt).astype(int)
        gt_f = write_lps(np.where((gt_arr >= 151) & (gt_arr <= 200), gt_arr, 0), gt)
        pf = write_lps(inst, ct)
        r = evaluate_case(pf, gt_f)
        os.unlink(pf); os.unlink(gt_f)
        row = []
        for k in keys:
            v = r.get(k, float("nan")); agg[k].append(v); row.append(f"{v:9.3f}")
        print(f"{cid:10s} " + " ".join(row), flush=True)

    print("\n=== held-out OOF 평균 (8지표) ===", flush=True)
    for k in keys:
        vals = [x for x in agg[k] if x == x]  # nan 제외
        print(f"  {k:10s} = {np.mean(vals):.4f}   (n={len(vals)})", flush=True)
    print("OOF8_DONE", flush=True)


if __name__ == "__main__":
    main()
