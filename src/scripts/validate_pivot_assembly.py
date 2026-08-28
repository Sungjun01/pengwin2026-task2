"""PIVOT-Assembly (v19) GT 검증.

전체 inference 파이프라인을 PIVOT_ASSEMBLY=1 로 GT 케이스(femur + pelvic)에 돌려
official metric 으로 채점. baseline(watershed femur 0.579 / legacy pelvic 0.763,
oracle 천장 0.96) 대비 얼마나 오르는지 = decode-fix 가 천장에 다가가는지 확인.

env(PIVOT_ASSEMBLY 등)는 호출 셸에서 export 한 값을 run_pipeline_v7 이 읽음.
"""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import run_pipeline_v7
from evaluation.pengwin_eval import evaluate_case

RES = Path("/home/schoaq/taehee/nnUNet_data/results")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
FEMUR = ["251", "252", "254", "256", "260", "263", "277"]
PELVIC = ["001", "005", "011", "017", "025"]


def lps(p):
    img = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f, img


def run_one(cid, lo, hi, tmp):
    outp = tmp / f"out_{cid}.mha"
    run_pipeline_v7(input_ct=RAW / cid / "image.mha", output_path=outp, nnunet_results=RES,
                    tile_step_size=1.0, use_pivot_form=False, use_d017_sacrum=False,
                    form_learn_mode=None, form_tau_intact=0.08,
                    min_volume_mm3=float(os.environ.get("PENGWIN_MIN_VOLUME_MM3", "500")))
    pa = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(outp)), "LPS"))
    pa = np.where((pa >= lo) & (pa <= hi), pa, 0)
    gimg = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
    ga = sitk.GetArrayFromImage(gimg).astype(int); ga = np.where((ga >= lo) & (ga <= hi), ga, 0)
    pf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    pimg = sitk.GetImageFromArray(pa.astype(np.uint16)); pimg.CopyInformation(gimg); sitk.WriteImage(pimg, pf)
    gf = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name; sitk.WriteImage(gimg, gf)
    r = evaluate_case(pf, gf); os.unlink(pf); os.unlink(gf)
    npred = len([v for v in np.unique(pa) if v != 0]); ngt = len([v for v in np.unique(ga) if v != 0])
    return r, npred, ngt


def main():
    tmp = Path(tempfile.mkdtemp(prefix="valasm_"))
    print(f"[validate] PIVOT_ASSEMBLY={os.environ.get('PIVOT_ASSEMBLY')} FEMUR_CHUNKS={os.environ.get('PIVOT_ASSEMBLY_FEMUR_CHUNKS')}", flush=True)
    fem, pel = [], []
    print("\n=== FEMUR (PIVOT-Assembly) — baseline watershed 0.579, 천장 0.964 ===", flush=True)
    for cid in FEMUR:
        r, np_, ng = run_one(cid, 151, 200, tmp); fem.append(r)
        print(f"  {cid}: f1={r['RQ_F']:.2f} (pred {np_}/gt {ng})", flush=True)
    print("\n=== PELVIC (PIVOT-Assembly) — baseline legacy 0.763, 천장 0.962 ===", flush=True)
    for cid in PELVIC:
        r, np_, ng = run_one(cid, 1, 150, tmp); pel.append(r)
        print(f"  {cid}: f1={r['RQ_F']:.2f} (pred {np_}/gt {ng})", flush=True)
    print(f"\n=== 평균 (ins_f1) ===", flush=True)
    print(f"  FEMUR  PIVOT-Assembly = {np.nanmean([r['RQ_F'] for r in fem]):.3f}  (watershed 0.579 / 천장 0.964)", flush=True)
    print(f"  PELVIC PIVOT-Assembly = {np.nanmean([r['RQ_F'] for r in pel]):.3f}  (legacy 0.763 / 천장 0.962)", flush=True)
    print("VAL_ASM_DONE", flush=True)


if __name__ == "__main__":
    main()
