"""femur decode 최종 bake-off (real-GT held-out 34) — watershed vs global-cc-v2.

각 케이스: D004 binary + D013 core/border 추론 → 두 decode 채점(8지표) + 레이더 분류.
  WS  = D004 binary → per_anatomy_watershed_instance (현 v18)
  GCC = global-cc-v2 (D004 글로벌CC + x-split + D013 골절분할)
net GCC > WS 면 v27 빌드.

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N python3 -m scripts.oof_femur_global_cc
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
from scripts.pivot_assembly import hybrid_femur_decode_global_cc
from scripts.oof_radar import print_radar_report, classify_signal
from evaluation.pengwin_eval import evaluate_case

FEMUR = (151, 200); MIN_VOL = 500.0
RES = Path("/home/schoaq/taehee/nnUNet_data/results")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset004_PENGWIN_FemurAnatomy/splits_final.json")
D004 = RES / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
D013 = RES / "Dataset013_PENGWIN_BorderCoreFemur" / "nnUNetTrainerBorderWeighted__nnUNetPlans__3d_fullres"


def init_pred(d):
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(d), use_folds=("all",), checkpoint_name="checkpoint_best.pth")
    return p


def wlps(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def nI(a):
    return len([v for v in np.unique(a) if 151 <= v <= 200])


def main():
    cids = [v.split("_")[-1] for v in json.load(open(SPLITS))[0]["val"]]
    p4 = init_pred(D004); p13 = init_pred(D013)
    tmp = Path(tempfile.mkdtemp(prefix="gcc_oof_"))
    aW = {k: [] for k in ["RQ_F", "HD95_F_mm", "ASSD_F_mm", "LDSC_F"]}
    aG = {k: [] for k in aW}
    radar_gcc = []
    print(f"[bake-off] watershed vs global-cc-v2  n={len(cids)}\n", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'nWS':>3s} {'nGCC':>4s} {'RQ_ws':>6s} {'RQ_gcc':>7s} {'ΔRQ':>6s}", flush=True)
    for cid in cids:
        ip = RAW / cid / "image.mha"
        if not ip.exists():
            continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(ip)), "LPS")
        gt = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(RAW/cid/"label.mha")), "LPS")).astype(int)
        gtN = len([v for v in np.unique(gt) if 151 <= v <= 200])
        sx, sy, sz = ct.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        cp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct, str(cp))
        gtf = wlps(np.where((gt>=151)&(gt<=200), gt, 0), ct)
        pr4 = predict_softmax(p4, cp, tmp/"o4"); d004 = (pr4[1] > 0.5)
        pr13 = predict_softmax(p13, cp, tmp/"o13"); sem = pr13.argmax(axis=0).astype(np.uint8)
        ws = per_anatomy_watershed_instance(d004.astype(np.uint8), 1, FEMUR, vv, MIN_VOL)
        gcc = hybrid_femur_decode_global_cc(d004, sem == 1, sem == 2, vv, spacing_zyx=spz,
                                            target_range=FEMUR, min_volume_mm3=MIN_VOL)
        rW = evaluate_case(wlps(ws, ct), gtf); rG = evaluate_case(wlps(gcc, ct), gtf)
        for k in aW:
            aW[k].append(rW.get(k, float("nan"))); aG[k].append(rG.get(k, float("nan")))
        radar_gcc.append({"case": cid, "gtN": gtN, "n_pred": nI(gcc),
                          "rq": rG["RQ_F"], "hd95": rG["HD95_F_mm"]})
        print(f"{cid:6s} {gtN:>3d} {nI(ws):>3d} {nI(gcc):>4d} {rW['RQ_F']:6.3f} {rG['RQ_F']:7.3f} "
              f"{rG['RQ_F']-rW['RQ_F']:+6.3f}", flush=True)

    print(f"\n=== net 평균 (n={len(aW['RQ_F'])}) ===")
    for k in aW:
        mw, mg = np.nanmean(aW[k]), np.nanmean(aG[k])
        print(f"  {k:10s} watershed={mw:7.3f}  global-cc={mg:7.3f}  Δ={mg-mw:+.3f}")
    print()
    print_radar_report(radar_gcc, mode="GLOBAL_CC_V2_OOF")
    print("GCC_OOF_DONE", flush=True)


if __name__ == "__main__":
    main()
