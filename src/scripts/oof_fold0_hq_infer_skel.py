"""Held-out fold_0 HQ-inference probe (budget-free ceiling of v18 weights).

v18 cripples PELVIC inference for the T4 10-min budget: use_mirroring=False,
use_gaussian=False, tile_step=1.0 (no overlap). This probes the gain from full HQ
inference on the SAME weights (Aseg_w40 fold_0), measured on held-out fold_0 (34
cases, no leakage). Decode is the canonical legacy border-core -> instances, so the
only change vs the baseline validation predictions is the inference quality.

Output: semantic 7-class .nii.gz into predictions/d016_fold0_hq, ready for
scripts.eval_heldout_bordercore (same decode/score as the 0.762 baseline).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from scripts.inference_v7_d015 import init_predictor  # noqa: E402

NNUNET_PRE = Path(os.environ["nnUNet_preprocessed"])
NNUNET_RES = Path(os.environ["nnUNet_results"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
DSET = "Dataset016_PENGWIN_BorderCore_MedialOrient"
MODEL_DIR = NNUNET_RES / DSET / "nnUNetTrainerSkeletonRecallFull__nnUNetPlans__3d_fullres"
IMAGES = NNUNET_RAW / DSET / "imagesTr"
OUT = _REPO / "predictions" / "d016_skel_fold0_hq"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(NNUNET_PRE / DSET / "splits_final.json"))
    val_cases = sorted(splits[0]["val"])
    print(f"[hq] fold_0 val cases: {len(val_cases)}", flush=True)

    # HQ inference on fold_0 weights: 3-axis mirroring TTA + gaussian + 0.5 overlap.
    predictor = init_predictor(
        str(MODEL_DIR),
        checkpoint="checkpoint_best.pth",
        use_mirroring=True,
        tile_step_size=0.5,
        use_gaussian=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_DIR),
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    for i, case in enumerate(val_cases):
        out_path = OUT / f"{case}.nii.gz"
        if out_path.exists():
            print(f"[hq] ({i+1}/{len(val_cases)}) {case} exists, skip", flush=True)
            continue
        channels = [str(IMAGES / f"{case}_{c:04d}.nii.gz") for c in range(5)]
        missing = [c for c in channels if not Path(c).exists()]
        if missing:
            print(f"[hq] {case} missing channels {missing}, skip", flush=True)
            continue
        tmp = OUT / f"_tmp_{case}"
        tmp.mkdir(parents=True, exist_ok=True)
        predictor.predict_from_files_sequential(
            list_of_lists_or_source_folder=[channels],
            output_folder_or_list_of_truncated_output_files=str(tmp),
            save_probabilities=False,
            overwrite=True,
            folder_with_segs_from_prev_stage=None,
        )
        produced = list(tmp.glob("*.nii.gz"))
        if not produced:
            print(f"[hq] {case} produced nothing!", flush=True)
            continue
        img = sitk.ReadImage(str(produced[0]))
        sitk.WriteImage(img, str(out_path))
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
        print(f"[hq] ({i+1}/{len(val_cases)}) {case} done", flush=True)

    print("HQ_FOLD0_DONE", flush=True)


if __name__ == "__main__":
    main()
