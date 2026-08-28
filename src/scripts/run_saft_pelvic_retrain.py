"""Retrain SAFT on the pelvic cases with the fixed pipeline.

Data is assembled directly from the existing nnUNet CT (imagesTr) + the raw
PENGWIN fragment GT (predictions/gt_instance), with no .mha conversion --
`load_raw_case` reads .nii.gz fine. The three hard eval cases (004/084/087) are
held out so post-training instance F1 is honest.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.train_saft import train_saft_streaming

REPO = Path(__file__).resolve().parents[1]
CT_DIR = Path("/home/schoaq/taehee/nnUNet_data/raw/Dataset001_PENGWIN_Anatomical/imagesTr")
GT_DIR = REPO / "predictions" / "gt_instance"
HELDOUT = {"004", "084", "087"}
OUTPUT_DIR = REPO / "logs" / "saft_pelvic_fixed_f48_p96"


def build_entries() -> list[tuple[str, Path, Path]]:
    entries: list[tuple[str, Path, Path]] = []
    for gt_path in sorted(GT_DIR.glob("PENGWIN_*.nii.gz")):
        case_id = gt_path.stem.replace("PENGWIN_", "").replace(".nii", "")
        if case_id in HELDOUT:
            continue
        ct_path = CT_DIR / f"PENGWIN_{case_id}_0000.nii.gz"
        if ct_path.exists():
            entries.append((case_id, ct_path, gt_path))
    return entries


def main() -> int:
    entries = build_entries()
    print(json.dumps({"event": "entries_built", "n_train": len(entries), "heldout": sorted(HELDOUT)}, sort_keys=True), flush=True)
    summary = train_saft_streaming(
        entries=entries,
        output_dir=OUTPUT_DIR,
        epochs=100,
        steps_per_epoch=100,
        patch_size=(96, 96, 96),
        backbone="swin_unetr",
        feature_channels=48,
        device="cuda",
        cache_size=170,  # cache all pelvic cases in RAM (~67GB) -> load each once, then pure compute
        lr=2e-4,
    )
    print(json.dumps({"event": "retrain_done", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
