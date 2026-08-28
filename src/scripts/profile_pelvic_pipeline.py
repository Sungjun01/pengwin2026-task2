"""Profile the pelvic pipeline buckets (D006 stage, D016 stage, decode+rest) to
find the TLE bottleneck. Hooks the two nnUNet calls + nnUNet preprocessing.

  CUDA_VISIBLE_DEVICES=2 python3 -m scripts.profile_pelvic_pipeline 099
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
TRAINSET = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset")

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
    PIVOT_ASSEMBLY="1", PIVOT_ASSEMBLY_FEMUR_CHUNKS="1",
    PENGWIN_MIN_VOLUME_MM3="500", PIVOT_V19_DECODE="0", PENGWIN_NNUNET_ON_DEVICE="1",
)

sys.path.insert(0, str(REPO))
from scripts.nnunet_preprocess_memory_patch import apply as _patch
_patch()

import scripts.inference_v7_d015 as inf

T = {}

# --- hook the two nnUNet stage calls ---
_orig_softmax = inf.predict_softmax
_orig_argmax = inf.predict_argmax_multichannel

def timed_softmax(*a, **k):
    t0 = time.perf_counter(); r = _orig_softmax(*a, **k); T["D006_total"] = time.perf_counter() - t0; return r

def timed_argmax(*a, **k):
    t0 = time.perf_counter(); r = _orig_argmax(*a, **k); T["D016_total"] = time.perf_counter() - t0; return r

inf.predict_softmax = timed_softmax
inf.predict_argmax_multichannel = timed_argmax

# --- hook nnUNet preprocess vs predict to split CPU-preprocess from GPU-inference ---
try:
    import nnunetv2.inference.predict_from_raw_data as pr
    PRED = pr.nnUNetPredictor
    _pp = PRED.predict_logits_from_preprocessed_data
    def timed_predict_logits(self, data, *a, **k):
        t0 = time.perf_counter(); r = _pp(self, data, *a, **k)
        T.setdefault("gpu_infer_sum", 0.0); T["gpu_infer_sum"] += time.perf_counter() - t0; return r
    PRED.predict_logits_from_preprocessed_data = timed_predict_logits
except Exception as e:
    print("[warn] could not hook predict_logits:", e)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "099"
    img = None
    for part in sorted(TRAINSET.glob("PENGWIN26_task1_2_train_part*")):
        p = part / case / "image.mha"
        if p.is_file(): img = p; break
    if img is None:
        sys.exit(f"no image for {case}")
    import SimpleITK as sitk
    sz = sitk.ReadImage(str(img)).GetSize()
    print(f"[profile] case {case} size(xyz)={sz}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{case}.mha"
        t0 = time.perf_counter()
        inf.run_pipeline_v7(input_ct=img, output_path=out, nnunet_results=MODELS,
                            tile_step_size=1.0, use_pivot_form=False, use_d017_sacrum=False,
                            form_learn_mode=None, form_tau_intact=0.08, min_volume_mm3=500.0)
        T["TOTAL"] = time.perf_counter() - t0
    d006 = T.get("D006_total", 0); d016 = T.get("D016_total", 0); tot = T["TOTAL"]
    gpu = T.get("gpu_infer_sum", 0)
    print("\n==== TIMING (s) ====")
    print(f"  D006 stage total      : {d006:7.1f}")
    print(f"  D016 stage total      : {d016:7.1f}")
    print(f"    of which GPU infer  : {gpu:7.1f}  (both stages; preprocess = stage_total - its infer)")
    print(f"  decode+assembly+write : {tot - d006 - d016:7.1f}")
    print(f"  TOTAL                 : {tot:7.1f}  ({tot/60:.1f} min)")
    print(f"  => CPU-preprocess+overhead (TOTAL - GPU infer) ≈ {tot - gpu:7.1f}")


if __name__ == "__main__":
    main()
