"""Grand-challenge.org algorithm entry point — single mha/tif in/out for PENGWIN 2026 Task 1.

Grand-challenge interface (PENGWIN 2026 Task 1 official).
    Input slug  : peripelvic-fracture-ct
        /input/images/peripelvic-fracture-ct/<UUID>.mha or .tif
    Output slug : peripelvic-fracture-ct-segmentation
        /output/images/peripelvic-fracture-ct-segmentation/<UUID>.mha or .tif

container 안에서 이 script 가 자동 실행. 모든 input mha/tif file 을 차례로 처리.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# project root 를 PYTHONPATH 에 추가 (container 안에서 import 가능하게)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.nnunet_preprocess_memory_patch import apply as _apply_nnunet_memory_patch

_apply_nnunet_memory_patch()

from scripts.inference_v7_d015 import run_pipeline_v7


INPUT_DIR = Path("/input/images/peripelvic-fracture-ct")
OUTPUT_DIR = Path("/output/images/peripelvic-fracture-ct-segmentation")
# 모델 위치: nnUNet_results env 우선(grand-challenge Algorithm Model 은 /opt/ml/model 로 추출).
# baked-in 컨테이너는 /opt/algorithm/nnUNet_results, model-tarball 방식은 /opt/ml/model.
RESULTS_DIR = Path(os.environ.get("nnUNet_results", "/opt/ml/model"))


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"input directory not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        list(INPUT_DIR.glob("*.mha")) + list(INPUT_DIR.glob("*.tif"))
    )
    if not input_files:
        sys.exit(f"no mha/tif files in {INPUT_DIR}")
    print(f"[setup] {len(input_files)} input cases")

    for in_path in input_files:
        out_path = OUTPUT_DIR / in_path.name
        print(f"[run] {in_path.name}")
        run_pipeline_v7(
            input_ct=in_path,
            output_path=out_path,
            nnunet_results=RESULTS_DIR,
            tile_step_size=1.0,        # T4 budget 위한 max non-overlap step
            use_pivot_form=False,
            use_d017_sacrum=os.environ.get("PIVOT_USE_D017_SACRUM", "0") == "1",
            form_learn_mode=(
                None
                if os.environ.get("PENGWIN_FORM_MODE", "") in ("", "legacy")
                else os.environ.get("PENGWIN_FORM_MODE")
            ),
            form_tau_intact=float(os.environ.get("PENGWIN_TAU_INTACT", "0.08")),
            min_volume_mm3=float(os.environ.get("PENGWIN_MIN_VOLUME_MM3", "500")),
        )

    print(f"[done] all {len(input_files)} cases written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
