"""v28 디코드 empirical bake-off (held-out D004 fold_0) — watershed vs v27-globalcc vs v28-monotone.

D004 = fold_0 (HELD-OUT, honest) / D013 = fold_all (memorized — 병합가이드는 낙관 주의).
핵심 확인: v28이 watershed보다 RQ_F를 올리되, v27이 터졌던 JAM-UNDER 케이스에서
HD95_F가 watershed처럼 *낮게* 유지되는가(= 단조-안전 경험적 확인).

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N python3 -m scripts.oof_femur_monotone
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
from scripts.pivot_assembly import hybrid_femur_decode_global_cc, monotone_safe_femur_decode
from evaluation.pengwin_eval import evaluate_case

FEMUR = (151, 200); MIN_VOL = 500.0
RES = Path("/home/schoaq/taehee/nnUNet_data/results")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset004_PENGWIN_FemurAnatomy/splits_final.json")
# ★ held-out fold_0 (honest) for D004 ★
D004 = RES / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"
D013 = RES / "Dataset013_PENGWIN_BorderCoreFemur" / "nnUNetTrainerBorderWeighted__nnUNetPlans__3d_fullres"


def init_pred(d, fold):
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(d), use_folds=(fold,), checkpoint_name="checkpoint_best.pth")
    return p


def wlps(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def nI(a):
    return len([v for v in np.unique(a) if 151 <= v <= 200])


def main():
    cids = [v.split("_")[-1] for v in json.load(open(SPLITS))[0]["val"]]
    p4 = init_pred(D004, "0"); p13 = init_pred(D013, "all")
    tmp = Path(tempfile.mkdtemp(prefix="mono_oof_"))
    KEYS = ["RQ_F", "HD95_F_mm", "ASSD_F_mm", "LDSC_F"]
    aW = {k: [] for k in KEYS}; aG = {k: [] for k in KEYS}; aM = {k: [] for k in KEYS}
    jam = []  # v28이 JAM-UNDER(HD95 폭발) 일으키는 케이스 추적
    print(f"[bake-off] watershed vs v27-gcc vs v28-monotone (D004 fold_0 held-out)  n={len(cids)}\n", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'nWS':>3s} {'nGCC':>4s} {'nMON':>4s} "
          f"{'RQ_ws':>6s} {'RQ_gcc':>6s} {'RQ_mon':>6s} {'H_ws':>6s} {'H_gcc':>6s} {'H_mon':>6s}", flush=True)
    for cid in cids:
        ip = RAW / cid / "image.mha"
        if not ip.exists():
            continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(ip)), "LPS")
        gt = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(RAW/cid/"label.mha")), "LPS")).astype(int)
        gtN = len([v for v in np.unique(gt) if 151 <= v <= 200])
        sx, sy, sz = ct.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        cp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct, str(cp))
        gtf = wlps(np.where((gt >= 151) & (gt <= 200), gt, 0), ct)
        pr4 = predict_softmax(p4, cp, tmp/"o4"); d004 = (pr4[1] > 0.5)
        pr13 = predict_softmax(p13, cp, tmp/"o13"); sem = pr13.argmax(axis=0).astype(np.uint8)
        ws = per_anatomy_watershed_instance(d004.astype(np.uint8), 1, FEMUR, vv, MIN_VOL)
        gcc = hybrid_femur_decode_global_cc(d004, sem == 1, sem == 2, vv, spacing_zyx=spz,
                                            target_range=FEMUR, min_volume_mm3=MIN_VOL)
        mon = monotone_safe_femur_decode(d004, sem == 1, sem == 2, vv, spacing_zyx=spz,
                                         target_range=FEMUR, min_volume_mm3=MIN_VOL)
        rW = evaluate_case(wlps(ws, ct), gtf)
        rG = evaluate_case(wlps(gcc, ct), gtf)
        rM = evaluate_case(wlps(mon, ct), gtf)
        for k in KEYS:
            aW[k].append(rW.get(k, float("nan")))
            aG[k].append(rG.get(k, float("nan")))
            aM[k].append(rM.get(k, float("nan")))
        # JAM-UNDER 감지: v28 HD95가 watershed보다 10mm+ 악화면 위험신호
        if rM["HD95_F_mm"] > rW["HD95_F_mm"] + 10.0:
            jam.append((cid, rW["HD95_F_mm"], rM["HD95_F_mm"]))
        print(f"{cid:6s} {gtN:>3d} {nI(ws):>3d} {nI(gcc):>4d} {nI(mon):>4d} "
              f"{rW['RQ_F']:6.3f} {rG['RQ_F']:6.3f} {rM['RQ_F']:6.3f} "
              f"{rW['HD95_F_mm']:6.1f} {rG['HD95_F_mm']:6.1f} {rM['HD95_F_mm']:6.1f}", flush=True)

    print(f"\n=== net 평균 (n={len(aW['RQ_F'])}) ===")
    print(f"{'metric':12s} {'watershed':>10s} {'v27-gcc':>9s} {'v28-mono':>9s} {'mono-ws':>8s}")
    for k in KEYS:
        mw, mg, mm = np.nanmean(aW[k]), np.nanmean(aG[k]), np.nanmean(aM[k])
        print(f"  {k:10s} {mw:10.3f} {mg:9.3f} {mm:9.3f} {mm-mw:+8.3f}")

    print(f"\n=== v28 JAM-UNDER 위험 케이스 (HD95 mono > ws+10mm) ===")
    if not jam:
        print("  없음 ✅ — v28이 어떤 케이스도 HD95 폭발시키지 않음 (단조-안전 경험적 확인)")
    else:
        for cid, hw, hm in jam:
            print(f"  ❌ {cid}: HD95 ws={hw:.1f} → mono={hm:.1f} (악화 {hm-hw:+.1f})")
    print("MONO_OOF_DONE", flush=True)


if __name__ == "__main__":
    main()
