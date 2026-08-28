"""Grand-challenge.org entry point for PENGWIN 2026 Task 1 using SAFT.

Interface (official PENGWIN 2026 Task 1 slugs):
    input : /input/images/peripelvic-fracture-ct/<UUID>.mha|.tif
    output: /output/images/peripelvic-fracture-ct-segmentation/<UUID>.mha|.tif

Runs the fixed SAFT model with the affinity-watershed instance decode. The
checkpoint is baked into the image at $SAFT_CHECKPOINT (default
/opt/algorithm/model/best.pt).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.infer_saft_case import infer_saft_case_from_paths

INPUT_DIR = Path("/input/images/peripelvic-fracture-ct")
OUTPUT_DIR = Path("/output/images/peripelvic-fracture-ct-segmentation")
CHECKPOINT = Path(os.environ.get("SAFT_CHECKPOINT", "/opt/algorithm/model/best.pt"))
MIN_VOXELS = int(os.environ.get("SAFT_MIN_VOXELS", "20"))
MERGE_THRESHOLD = float(os.environ.get("SAFT_MERGE_THRESHOLD", "0.5"))
DECODE = os.environ.get("SAFT_DECODE", "affinity")


def main() -> int:
    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")
    if not INPUT_DIR.exists():
        sys.exit(f"input directory not found: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list(INPUT_DIR.glob("*.mha")) + list(INPUT_DIR.glob("*.tif")))
    if not input_files:
        sys.exit(f"no mha/tif files in {INPUT_DIR}")
    print(f"[setup] checkpoint={CHECKPOINT} decode={DECODE} min_voxels={MIN_VOXELS} cases={len(input_files)}", flush=True)

    for in_path in input_files:
        out_path = OUTPUT_DIR / in_path.name
        print(f"[run] {in_path.name}", flush=True)
        infer_saft_case_from_paths(
            checkpoint_path=CHECKPOINT,
            image_path=in_path,
            output_path=out_path,
            min_voxels=MIN_VOXELS,
            device="auto",
            decode=DECODE,
            merge_threshold=MERGE_THRESHOLD,
        )
    print(f"[done] {len(input_files)} cases written to {OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
