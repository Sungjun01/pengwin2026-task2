"""Build nnUNet raw datasets D012 (pelvic) + D013 (femur) for border-core training.

Pelvic D012 reuses D010's 4-channel images (CT + offset x3) verbatim (symlink);
femur D013 reuses D004's CT image. Only the labels change: PENGWIN instance GT
-> seam border-core joint label map (see border_core_conversion.py).

Usage:
    python scripts/build_border_core_datasets.py [--limit N] [--anatomy pelvic|femur|both]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.border_core_conversion import convert_to_border_core, PELVIC_ANATOMY, FEMUR_ANATOMY

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
GT_INSTANCE = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance")
RAW_TRAIN = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")

PELVIC_LABELS = {"background": 0, "sacrum_core": 1, "sacrum_border": 2,
                 "leftHip_core": 3, "leftHip_border": 4, "rightHip_core": 5, "rightHip_border": 6}
FEMUR_LABELS = {"background": 0, "femur_core": 1, "femur_border": 2}


def case_ids(src_imagesTr: Path) -> list[str]:
    return sorted({f.name[:11] for f in src_imagesTr.glob("*_0000.nii.gz")})


def build(name: str, src_ds: str, n_channels: int, anatomy, labels, instance_source, limit=None):
    src = NRAW / src_ds / "imagesTr"
    dst = NRAW / name
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cids = case_ids(src)
    if limit:
        cids = cids[:limit]
    print(f"[{name}] {len(cids)} cases from {src_ds}")
    for i, cid in enumerate(cids):
        # symlink image channels (absolute targets)
        for c in range(n_channels):
            link = dst / "imagesTr" / f"{cid}_{c:04d}.nii.gz"
            target = (src / f"{cid}_{c:04d}.nii.gz").resolve()
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        # convert instance GT -> border-core, write with image geometry
        inst_path = instance_source(cid)
        inst = sitk.ReadImage(str(inst_path))
        arr = sitk.GetArrayFromImage(inst)
        bc = convert_to_border_core(arr, anatomy, t=2)
        ref = sitk.ReadImage(str(src / f"{cid}_0000.nii.gz"))
        assert ref.GetSize() == inst.GetSize(), f"{cid} geometry mismatch {ref.GetSize()} vs {inst.GetSize()}"
        out = sitk.GetImageFromArray(bc)
        out.CopyInformation(ref)
        sitk.WriteImage(out, str(dst / "labelsTr" / f"{cid}.nii.gz"), useCompression=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cids)}")
    ds = {"channel_names": {str(c): ("CT" if c == 0 else f"offset{c}") for c in range(n_channels)},
          "labels": labels, "numTraining": len(cids), "file_ending": ".nii.gz"}
    (dst / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[{name}] done. dataset.json: {n_channels}ch, {len(labels)} classes, {len(cids)} training")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--anatomy", choices=["pelvic", "femur", "both"], default="both")
    a = ap.parse_args()
    if a.anatomy in ("pelvic", "both"):
        build("Dataset012_PENGWIN_BorderCorePelvic", "Dataset010_PENGWIN_CentroidOnly_Predicted",
              4, PELVIC_ANATOMY, PELVIC_LABELS,
              lambda cid: GT_INSTANCE / f"{cid}.nii.gz", limit=a.limit)
    if a.anatomy in ("femur", "both"):
        build("Dataset013_PENGWIN_BorderCoreFemur", "Dataset004_PENGWIN_FemurAnatomy",
              1, FEMUR_ANATOMY, FEMUR_LABELS,
              lambda cid: RAW_TRAIN / str(int(cid.split("_")[1])) / "label.mha", limit=a.limit)


if __name__ == "__main__":
    main()
