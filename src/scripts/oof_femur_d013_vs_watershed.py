"""femur decode 비교 — watershed(D004) vs d013 border-core, 같은 34 케이스 8지표.

watershed가 깨끗한 binary를 과분할하는 게 확정(report 88)됐으니, 대안 decode인
d013 border-core(모델이 골절면=border를 학습 → 거기서만 자름)가 인스턴스(RQ_F)를
올리는지 검증.

  arm A = D004 fold_all → binary → per_anatomy_watershed_instance
  arm B = D013 fold_all → argmax(bg/core/border) → border_core_to_instances

⚠️ 둘 다 fold_all(memorized)이라 절대값은 낙관적. *상대 비교*(d013-watershed)가 신호.
honest 절대값은 추후 d013 fold_0 학습 필요.

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 -m scripts.oof_femur_d013_vs_watershed
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
from scripts.pivot_form_postproc import border_core_to_instances
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
    tmp = Path(tempfile.mkdtemp(prefix="d013cmp_"))
    K = ["IoU_A", "HD95_A_mm", "ASSD_A_mm", "RQ_F", "HD95_F_mm", "ASSD_F_mm", "LDSC_F"]
    aW = {k: [] for k in K}; aD = {k: [] for k in K}
    print(f"[d013 vs watershed] n={len(cids)}  (둘 다 fold_all memorized)\n", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'nWS':>3s} {'nD13':>4s} {'RQ_ws':>6s} {'RQ_d13':>7s} {'ΔRQ':>6s}  flag", flush=True)
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
        # A: watershed
        pr4 = predict_softmax(p4, cp, tmp/"o4")
        ws = per_anatomy_watershed_instance((pr4[1] > 0.5).astype(np.uint8), 1, FEMUR, vv, MIN_VOL)
        rW = evaluate_case(wlps(ws, ct), gtf)
        # B: d013 border-core
        pr13 = predict_softmax(p13, cp, tmp/"o13")
        sem = pr13.argmax(axis=0).astype(np.uint8)
        d13 = border_core_to_instances(sem, 1, 2, FEMUR, vv, min_volume_mm3=MIN_VOL,
                                       form_cfg=None, spacing_zyx=spz)
        rD = evaluate_case(wlps(d13, ct), gtf)
        for k in K:
            aW[k].append(rW.get(k, float("nan"))); aD[k].append(rD.get(k, float("nan")))
        dRQ = rD['RQ_F'] - rW['RQ_F']
        flag = "d13⬆" if dRQ > 0.01 else ("d13⬇" if dRQ < -0.01 else "~")
        print(f"{cid:6s} {gtN:>3d} {nI(ws):>3d} {nI(d13):>4d} {rW['RQ_F']:6.3f} {rD['RQ_F']:7.3f} {dRQ:+6.3f}  {flag}", flush=True)

    print(f"\n=== 평균 (n={len(aW['RQ_F'])}, 둘 다 memorized) ===", flush=True)
    print(f"{'지표':10s} {'watershed':>10s} {'d013':>10s} {'Δ(d13-ws)':>10s}")
    for k in K:
        mw = np.nanmean(aW[k]); md = np.nanmean(aD[k])
        print(f"  {k:10s} {mw:10.3f} {md:10.3f} {md-mw:+10.3f}")
    print("D013CMP_DONE", flush=True)


if __name__ == "__main__":
    main()
