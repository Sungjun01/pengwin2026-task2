"""femur 기준선(D004+watershed)을 **Dataset023 fold_0 val과 동일한 34케이스**에 채점.
D004는 fold_all(그 케이스 학습에 봄)이라 optimistic baseline — SDT(held-out)가 이걸 이기면 강한 신호.
oof_femur_8metric.py와 동일 로직, splits만 Dataset023 + fold=all.
용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 -m scripts.oof_femur_baseline_d023
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

FEMUR = (151, 200); MIN_VOL = 500.0
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset023_PENGWIN_SDT_Femur/splits_final.json")
CKPT_DIR = "/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy/nnUNetTrainer__nnUNetPlans__3d_fullres"
FOLD = "all"


def main():
    val = json.load(open(SPLITS))[0]["val"]
    cids = [v.split("_")[-1] for v in val]
    print(f"[BASE-OOF] ckpt={Path(CKPT_DIR).name} fold={FOLD} (optimistic, 같은 34케이스)", flush=True)
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(CKPT_DIR, use_folds=(FOLD,), checkpoint_name="checkpoint_final.pth")
    tmp = Path(tempfile.mkdtemp(prefix="oofbase_"))
    keys = ["IoU_A", "HD95_A_mm", "ASSD_A_mm", "IoU_F", "HD95_F_mm", "ASSD_F_mm", "RQ_F", "LDSC_F"]
    agg = {k: [] for k in keys}
    print(f"{'case':10s} " + " ".join(f"{k:>9s}" for k in keys), flush=True)
    for cid in cids:
        img_p = RAW / cid / "image.mha"
        if not img_p.exists():
            print(f"{cid}: skip", flush=True); continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(img_p)), "LPS")
        gt = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        sx, sy, sz = ct.GetSpacing(); vv = float(sx * sy * sz)
        ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct, str(ct_path))
        probs = predict_softmax(p, ct_path, tmp / "out")
        fem_bin = (probs[1].astype(np.float32) > 0.5).astype(np.uint8)
        inst = per_anatomy_watershed_instance(fem_bin, 1, FEMUR, vv, MIN_VOL)
        gt_arr = sitk.GetArrayFromImage(gt).astype(int)
        gimg = sitk.GetImageFromArray(np.where((gt_arr >= 151) & (gt_arr <= 200), gt_arr, 0).astype(np.uint16)); gimg.CopyInformation(gt)
        pimg = sitk.GetImageFromArray(inst.astype(np.uint16)); pimg.CopyInformation(ct)
        gf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name; sitk.WriteImage(gimg, gf)
        pf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name; sitk.WriteImage(pimg, pf)
        r = evaluate_case(pf, gf); os.unlink(pf); os.unlink(gf)
        row = []
        for k in keys:
            v = r.get(k, float("nan")); agg[k].append(v); row.append(f"{v:9.3f}")
        print(f"{cid:10s} " + " ".join(row), flush=True)
    print("\n=== BASELINE held-out(optimistic) 평균 (8지표) ===", flush=True)
    for k in keys:
        vals = [x for x in agg[k] if x == x]
        print(f"  {k:10s} = {np.mean(vals):.4f}   (n={len(vals)})", flush=True)
    print("BASE_OOF8_DONE", flush=True)


if __name__ == "__main__":
    main()
