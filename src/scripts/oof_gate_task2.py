"""Task2 OOF gate: run v19 once per case, score v5 vs v6 click postproc on the
SAME v19 instance map with the official pengwin_eval. Confirms the coordinate fix
is a net win / not a regression.

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 -m scripts.oof_gate_task2 --cases 099,200 --out /tmp/gate_g3.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
TRAINSET = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset")
CLICKS = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/task2/PENGWIN26_task2_train_clicks")
STRATEGIES = ["center_of_mass", "euclidean_distance_transform", "uniformly_sampled", "boundary_internal_margin"]

# Mirror Dockerfile.task2 ENV so the host run matches the container exactly.
os.environ.setdefault("nnUNet_results", str(MODELS))
os.environ.setdefault("nnUNet_raw", str(REPO / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(REPO / "nnUNet_preprocessed"))
os.environ.update(
    PIVOT_PELVIC_DATASET="Dataset016_PENGWIN_BorderCore_MedialOrient",
    PIVOT_PELVIC_TRAINER_PLANS="nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres",
    PIVOT_PELVIC_FOLDS="all",
    PIVOT_FEMUR_DATASET="Dataset004_PENGWIN_FemurAnatomy",
    PIVOT_FEMUR_TRAINER_PLANS="nnUNetTrainer__nnUNetPlans__3d_fullres",
    PIVOT_FEMUR_INSTANCE_METHOD="watershed",
    PIVOT_ASSEMBLY="1",
    PIVOT_ASSEMBLY_FEMUR_CHUNKS="1",
    PIVOT_ASSEMBLY_MAX_SLIVER_MM3="2000",
    PIVOT_ASSEMBLY_MIN_VOL_RATIO="5",
    PIVOT_ASSEMBLY_DILATION_ITERS="1",
    PIVOT_ASSEMBLY_MAX_DIST_MM="30",
    PENGWIN_MIN_VOLUME_MM3="500",
    PIVOT_V19_DECODE="0",
    PENGWIN_NNUNET_ON_DEVICE="1",
)

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.nnunet_preprocess_memory_patch import apply as _patch  # noqa: E402

_patch()
from scripts.inference_v7_d015 import run_pipeline_v7  # noqa: E402
from evaluation.pengwin_eval import evaluate_case  # noqa: E402
from scripts import click_guided_postproc as v6  # noqa: E402
from scripts import click_guided_postproc_v5 as v5  # noqa: E402

MIN_CC = 1000.0


def find_image_label(case: str):
    for part in sorted(TRAINSET.glob("PENGWIN26_task1_2_train_part*")):
        img = part / case / "image.mha"
        lab = part / case / "label.mha"
        if img.is_file() and lab.is_file():
            return img, lab
    return None, None


def score_one(case: str, tmp: Path) -> dict:
    img_p, lab_p = find_image_label(case)
    if img_p is None:
        return {"case": case, "error": "no image/label"}
    inst_p = tmp / f"{case}_v19.mha"
    run_pipeline_v7(
        input_ct=img_p, output_path=inst_p, nnunet_results=MODELS,
        tile_step_size=1.0, use_pivot_form=False, use_d017_sacrum=False,
        form_learn_mode=None, form_tau_intact=0.08, min_volume_mm3=500.0,
    )
    inst_img = sitk.ReadImage(str(inst_p))
    inst_arr = sitk.GetArrayFromImage(inst_img)
    rec = {"case": case, "n_inst_ids": int(len(set(inst_arr[inst_arr > 0].tolist()))), "strategies": {}}
    for strat in STRATEGIES:
        cj = CLICKS / strat / case / "peripelvic-fragment-clicks.json"
        if not cj.is_file():
            continue
        entry = {}
        for tag, mod in (("v5", v5), ("v6", v6)):
            try:
                clicks = mod.parse_clicks(cj)
                out = mod.relabel_from_clicks(inst_arr, clicks)
                pp = tmp / f"{case}_{strat}_{tag}.mha"
                oi = sitk.GetImageFromArray(out.astype(np.uint8))
                oi.CopyInformation(inst_img)
                sitk.WriteImage(sitk.Cast(oi, sitk.sitkUInt8), str(pp), useCompression=True)
                m = evaluate_case(pp, lab_p, min_cc_mm3=MIN_CC)
                entry[tag] = m
            except Exception as e:
                entry[tag] = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
        rec["strategies"][strat] = entry
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    results = []
    with tempfile.TemporaryDirectory(prefix="oofgate_") as tmp_s:
        tmp = Path(tmp_s)
        for case in cases:
            print(f"[gate] === case {case} ===", flush=True)
            try:
                r = score_one(case, tmp)
            except Exception as e:
                r = {"case": case, "error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            results.append(r)
            Path(args.out).write_text(json.dumps(results, indent=2))
            print(f"[gate] case {case} done -> {args.out}", flush=True)
    print(f"[gate] all done: {len(results)} cases -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
