"""Run v19 on a case, then time the click postproc sub-stages on the REAL v19
instance map (not GT) to locate the wall-time hog.

  CUDA_VISIBLE_DEVICES=2 python3 -m scripts.profile_task2_postproc 084
"""
from __future__ import annotations
import os, sys, time, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
TRAIN = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset")
CLICKS = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/task2/PENGWIN26_task2_train_clicks")

os.environ.setdefault("nnUNet_results", str(MODELS))
os.environ.update(
    PIVOT_PELVIC_DATASET="Dataset016_PENGWIN_BorderCore_MedialOrient",
    PIVOT_PELVIC_TRAINER_PLANS="nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres",
    PIVOT_PELVIC_FOLDS="all", PIVOT_FEMUR_DATASET="Dataset004_PENGWIN_FemurAnatomy",
    PIVOT_FEMUR_TRAINER_PLANS="nnUNetTrainer__nnUNetPlans__3d_fullres",
    PIVOT_FEMUR_INSTANCE_METHOD="watershed", PIVOT_ASSEMBLY="1", PIVOT_ASSEMBLY_FEMUR_CHUNKS="1",
    PENGWIN_MIN_VOLUME_MM3="500", PIVOT_V19_DECODE="0", PENGWIN_NNUNET_ON_DEVICE="1",
    PIVOT_RESAMPLE_WORKERS="8",
)
sys.path.insert(0, str(REPO))
from scripts.nnunet_preprocess_memory_patch import apply as _p; _p()
from scripts.inference_v7_d015 import run_pipeline_v7
import scripts.click_guided_postproc as cgp
import numpy as np, SimpleITK as sitk

# instrument sub-stages
T = {"absorb": 0.0, "split": 0.0}
_abs = cgp._absorb_residual
_spl = cgp._split_mask_by_clicks
def t_abs(*a, **k):
    t0 = time.perf_counter(); r = _abs(*a, **k); T["absorb"] += time.perf_counter() - t0; return r
def t_spl(*a, **k):
    t0 = time.perf_counter(); r = _spl(*a, **k); T["split"] += time.perf_counter() - t0; return r
cgp._absorb_residual = t_abs
cgp._split_mask_by_clicks = t_spl


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "084"
    img = next(p / case / "image.mha" for p in TRAIN.glob("PENGWIN26_task1_2_train_part*") if (p/case/"image.mha").is_file())
    with tempfile.TemporaryDirectory() as tmp:
        v19 = Path(tmp) / f"{case}.mha"
        t0 = time.perf_counter()
        run_pipeline_v7(input_ct=img, output_path=v19, nnunet_results=MODELS, tile_step_size=1.0,
                        use_pivot_form=False, use_d017_sacrum=False, form_learn_mode=None,
                        form_tau_intact=0.08, min_volume_mm3=500.0)
        t_pipe = time.perf_counter() - t0
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(v19)))
        n_cc = len(set(arr[arr > 0].tolist()))
        clicks = cgp.parse_clicks(CLICKS / "center_of_mass" / case / "peripelvic-fragment-clicks.json")
        t0 = time.perf_counter()
        out = cgp.relabel_from_clicks(arr, clicks)
        t_relabel = time.perf_counter() - t0
    print(f"\n==== {case} (voxels {arr.size/1e6:.0f}M, v19 labels {n_cc}, clicks {len(clicks)}) ====")
    print(f"  pipeline run_pipeline_v7 : {t_pipe:7.1f}s")
    print(f"  relabel_from_clicks TOTAL: {t_relabel:7.1f}s")
    print(f"    _absorb_residual sum   : {T['absorb']:7.1f}s")
    print(f"    _split_mask sum        : {T['split']:7.1f}s")
    print(f"    resolution+rest        : {t_relabel - T['absorb'] - T['split']:7.1f}s")
    print(f"  => out fragments {len(set(out[out>0].tolist()))}")


if __name__ == "__main__":
    main()
