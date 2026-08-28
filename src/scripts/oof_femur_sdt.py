"""femur fold_0 held-out OOF — SDT 모델 채점 (oof_femur_8metric의 SDT 변형).
Dataset023(SDT 에너지 11-class) 예측 → argmax(bin) → sdt_instances decode → evaluate_case 8지표.
oof_femur_8metric.py(D004+watershed=챔피언 기준선)와 *같은 평가*로 비교 → SDT의 표면 이득 검증.
용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 -m scripts.oof_femur_sdt
"""
from __future__ import annotations
import sys, os, json, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scripts.inference_v7_d015 import predict_softmax
from scripts.sdt_decode import sdt_instances
from evaluation.pengwin_eval import evaluate_case

RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset023_PENGWIN_SDT_Femur/splits_final.json")
CKPT_DIR = os.environ.get("PENGWIN_OOF_CKPT",
    "/home/schoaq/taehee/nnUNet_data/results/Dataset023_PENGWIN_SDT_Femur/nnUNetTrainer__nnUNetPlans__3d_fullres")
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
    print(f"[SDT-OOF] ckpt={Path(CKPT_DIR).name} fold={FOLD}  held-out n={len(cids)}", flush=True)
    pred = init_fold_predictor(CKPT_DIR, FOLD)
    tmp = Path(tempfile.mkdtemp(prefix="oofsdt_"))
    keys = ["IoU_A", "HD95_A_mm", "ASSD_A_mm", "IoU_F", "HD95_F_mm", "ASSD_F_mm", "RQ_F", "LDSC_F"]
    agg = {k: [] for k in keys}
    print(f"{'case':10s} " + " ".join(f"{k:>9s}" for k in keys), flush=True)
    for cid in cids:
        img_p = RAW / cid / "image.mha"
        if not img_p.exists():
            print(f"{cid}: image 없음 skip", flush=True); continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(img_p)), "LPS")
        gt = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct, str(ct_path))
        probs = predict_softmax(pred, ct_path, tmp / "out")   # (11, Z, Y, X)
        sem = probs.argmax(0).astype(np.int32)                # SDT 에너지 bin 0..10
        inst = sdt_instances(sem)                             # femur instances (151+)
        gt_arr = sitk.GetArrayFromImage(gt).astype(int)
        gt_f = write_lps(np.where((gt_arr >= 151) & (gt_arr <= 200), gt_arr, 0), gt)
        pf = write_lps(inst, ct)
        r = evaluate_case(pf, gt_f)
        os.unlink(pf); os.unlink(gt_f)
        row = []
        for k in keys:
            v = r.get(k, float("nan")); agg[k].append(v); row.append(f"{v:9.3f}")
        print(f"{cid:10s} " + " ".join(row), flush=True)

    print("\n=== SDT held-out OOF 평균 (8지표) ===", flush=True)
    for k in keys:
        vals = [x for x in agg[k] if x == x]
        print(f"  {k:10s} = {np.mean(vals):.4f}   (n={len(vals)})", flush=True)
    print("SDT_OOF8_DONE", flush=True)


if __name__ == "__main__":
    main()
