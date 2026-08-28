"""femur over-split 원인 분리 — *완전 학습* D004(1000ep fold_all)로 진단.

질문: 골절 femur over-split(2 GT → 6~9 chunk)이
  (a) binary 자체가 깨짐(모델/under-training) 인가, vs
  (b) watershed가 *깨끗한* binary를 쪼갬(decode) 인가?

각 케이스: D004 fold_all → binary
  · n_CC      = binary의 26-conn 연결성분 수 (≥500mm³)   ← 깨끗하면 ~gtN
  · n_ws      = per_anatomy_watershed_instance 후 chunk 수 ← watershed 과분할 척도
  · gtN       = GT femur 조각 수
n_CC≈gtN 인데 n_ws≫gtN 이면 → watershed가 원흉(decode 문제).
n_CC≫gtN 이면 → binary 깨짐(모델 문제).

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 -m scripts.femur_oversplit_diag 284 286 291 293 302 254 259
"""
from __future__ import annotations
import sys, os, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label as cc_label, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scripts.inference_v7_d015 import predict_softmax, per_anatomy_watershed_instance

ST = generate_binary_structure(3, 3)
FEMUR = (151, 200); MIN_VOL = 500.0
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
# 완전 학습 deployed D004 (1000ep fold_all)
D004 = Path("/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy/nnUNetTrainer__nnUNetPlans__3d_fullres")


def main():
    cids = sys.argv[1:] or ["284", "286", "291", "293", "302", "254", "259"]
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(D004), use_folds=("all",), checkpoint_name="checkpoint_best.pth")
    tmp = Path(tempfile.mkdtemp(prefix="osdiag_"))
    print(f"[over-split diag] 완전학습 D004 fold_all (1000ep)\n", flush=True)
    print(f"{'case':6s} {'gtN':>4s} {'n_CC(binary)':>13s} {'n_watershed':>12s}  판정", flush=True)
    for cid in cids:
        ip = RAW / cid / "image.mha"
        if not ip.exists():
            print(f"{cid}: image 없음 skip", flush=True); continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(ip)), "LPS")
        gt = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(RAW/cid/"label.mha")), "LPS")).astype(int)
        gtN = len([v for v in np.unique(gt) if 151 <= v <= 200])
        sx, sy, sz = ct.GetSpacing(); vv = float(sx*sy*sz)
        cp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct, str(cp))
        probs = predict_softmax(p, cp, tmp/"o")
        fem = (probs[1].astype(np.float32) > 0.5)
        # binary 연결성분 (≥500mm³)
        lab, n = cc_label(fem, structure=ST)
        n_cc = sum(1 for c in range(1, n+1) if (lab == c).sum()*vv >= MIN_VOL)
        # watershed chunk
        ws = per_anatomy_watershed_instance(fem.astype(np.uint8), 1, FEMUR, vv, MIN_VOL)
        n_ws = len([v for v in np.unique(ws) if 151 <= v <= 200])
        verdict = ("watershed 과분할(decode)" if n_cc <= gtN+1 and n_ws > gtN+1
                   else "binary 깨짐(모델)" if n_cc > gtN+1
                   else "정상")
        print(f"{cid:6s} {gtN:>4d} {n_cc:>13d} {n_ws:>12d}  {verdict}", flush=True)
    print("OSDIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
