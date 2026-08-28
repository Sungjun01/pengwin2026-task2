"""oracle-decode 천장 테스트: 병목이 decode 인지 semantic 모델인지 가린다.

각 케이스에서 모델의 semantic 예측 foreground 를 GT instance 로 *완벽 분리* (oracle):
  oracle = where(predicted_foreground, GT_instance_label, 0)
→ 우리가 예측한 foreground 를 perfect 하게 갈랐을 때의 ins_f1 천장.

이 천장 vs 실제 휴리스틱 decode 점수 비교:
  - 천장 높음(~0.9+) & 휴리스틱 낮음 → decode 가 병목 (새 모델 불필요, decode 고치면 됨)
  - 천장도 낮음 → semantic 모델이 fragment 를 못 잡음 (새 알고리즘 정당)

femur: D004 binary foreground / GT(151-200).  pelvic: D016 core(1,3,5) foreground / GT(1-150).
"""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import (
    init_predictor, predict_softmax, predict_argmax_multichannel,
    per_anatomy_watershed_instance, compute_d015_offset_channels, save_temp_channel,
)
from scripts.pivot_form_postproc import border_core_to_instances
from evaluation.pengwin_eval import evaluate_case

RES = Path("/home/schoaq/taehee/nnUNet_data/results")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
D004 = RES / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
D013 = RES / "Dataset013_PENGWIN_BorderCoreFemur" / "nnUNetTrainerBorderWeighted__nnUNetPlans__3d_fullres"
D006 = RES / "Dataset006_PENGWIN_SacrumOnly" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
D016 = RES / "Dataset016_PENGWIN_BorderCore_MedialOrient" / "nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres"
MIN_VOL = 500.0
FEMUR_CASES = ["251", "252", "254", "256", "260", "263", "277"]
PELVIC_CASES = ["001", "005", "011", "017", "025"]  # pelvic train cases (GT in raw_data)


def lps_arr(path):
    img = sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")
    return img, sitk.GetArrayFromImage(img)


def wtmp(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def score(pred_arr, gt_arr, ref):
    pf, gf = wtmp(pred_arr, ref), wtmp(gt_arr, ref)
    r = evaluate_case(pf, gf); os.unlink(pf); os.unlink(gf); return r


def main():
    tmp = Path(tempfile.mkdtemp(prefix="oracle_"))
    print("[oracle] predictors ...", flush=True)
    p004 = init_predictor(str(D004), tile_step_size=1.0)
    p013 = init_predictor(str(D013), tile_step_size=1.0)
    p006 = init_predictor(str(D006), tile_step_size=1.0)
    p016 = init_predictor(str(D016), tile_step_size=1.0)

    rows = []
    # ---------- FEMUR ----------
    print("\n=== FEMUR (D004 binary foreground) ===", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'oracle_f1':>10s} {'watershed_f1':>13s} {'D013_f1':>9s}", flush=True)
    for cid in FEMUR_CASES:
        ct_img, _ = lps_arr(RAW / cid / "image.mha")
        gt_img, gt = lps_arr(RAW / cid / "label.mha"); gt = gt.astype(int)
        gt_f = np.where((gt >= 151) & (gt <= 200), gt, 0)
        gtN = len([v for v in np.unique(gt_f) if v != 0])
        sx, sy, sz = ct_img.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        ctp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct_img, str(ctp))
        fem_bin = (predict_softmax(p004, ctp, tmp/"d004")[1] > 0.5).astype(np.uint8)
        sem013 = predict_softmax(p013, ctp, tmp/"d013").argmax(0).astype(np.uint8)
        oracle = np.where(fem_bin > 0, gt_f, 0)                       # perfect separation of pred FG
        wat = per_anatomy_watershed_instance(fem_bin, 1, (151,200), vv, MIN_VOL)
        d13 = border_core_to_instances(sem013, 1, 2, (151,200), vv, min_volume_mm3=MIN_VOL, form_cfg=None, spacing_zyx=spz)
        ro = score(oracle, gt_f, ct_img); rw = score(wat, gt_f, ct_img); rd = score(d13, gt_f, ct_img)
        rows.append(("femur", ro, rw, rd))
        print(f"{cid:6s} {gtN:>3d} {ro['RQ_F']:>10.2f} {rw['RQ_F']:>13.2f} {rd['RQ_F']:>9.2f}", flush=True)

    # ---------- PELVIC ----------
    print("\n=== PELVIC (D016 core foreground) ===", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'oracle_f1':>10s} {'legacy_f1':>10s}", flush=True)
    for cid in PELVIC_CASES:
        ct0, _ = lps_arr(RAW / cid / "image.mha")
        gt_img, gt = lps_arr(RAW / cid / "label.mha"); gt = gt.astype(int)
        gt_p = np.where((gt >= 1) & (gt <= 150), gt, 0)
        gtN = len([v for v in np.unique(gt_p) if v != 0])
        sx, sy, sz = ct0.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        ctp = tmp/"pct_0000.nii.gz"; sitk.WriteImage(ct0, str(ctp))
        # sacrum (D006) -> 5ch -> D016 pelvic semantic
        psac = predict_softmax(p006, ctp, tmp/"psac")[1]
        sacm = (psac > 0.5)
        offs = compute_d015_offset_channels(sacm, ct0.GetSpacing())
        if offs is None:
            offs = np.zeros((3,)+psac.shape, np.float32); side = np.zeros(psac.shape, np.float32)
        else:
            side = (offs[0] > 0).astype(np.float32)
        chans = [ctp]
        for i, ch in enumerate(offs, 1):
            pth = tmp/f"pct_{i:04d}.nii.gz"; save_temp_channel(ch, ct0, pth); chans.append(pth)
        sp = tmp/"pct_0004.nii.gz"; save_temp_channel(side, ct0, sp); chans.append(sp)
        sem016 = predict_argmax_multichannel(p016, chans, tmp/"d016")
        full_fg = sem016 > 0                                          # pelvic 예측 전체(core+border)
        oracle = np.where(full_fg, gt_p, 0)
        # legacy decode per anatomy
        leg = np.zeros(sem016.shape, np.uint16)
        for core, bord, rng in [(1,2,(1,50)), (3,4,(51,100)), (5,6,(101,150))]:
            leg += border_core_to_instances(sem016, core, bord, rng, vv, min_volume_mm3=MIN_VOL, form_cfg=None, spacing_zyx=spz)
        ro = score(oracle, gt_p, ct0); rl = score(leg, gt_p, ct0)
        rows.append(("pelvic", ro, rl, None))
        print(f"{cid:6s} {gtN:>3d} {ro['RQ_F']:>10.2f} {rl['RQ_F']:>10.2f}", flush=True)

    print("\n=== 평균 (RQ_F = ins_f1) ===", flush=True)
    fem = [r for r in rows if r[0]=="femur"]; pel = [r for r in rows if r[0]=="pelvic"]
    if fem:
        print(f"  FEMUR  oracle={np.nanmean([r[1]['RQ_F'] for r in fem]):.3f}  watershed={np.nanmean([r[2]['RQ_F'] for r in fem]):.3f}  D013={np.nanmean([r[3]['RQ_F'] for r in fem]):.3f}", flush=True)
    if pel:
        print(f"  PELVIC oracle={np.nanmean([r[1]['RQ_F'] for r in pel]):.3f}  legacy={np.nanmean([r[2]['RQ_F'] for r in pel]):.3f}", flush=True)
    print("ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
