"""Build Dataset014_PENGWIN_BorderCore_SideMask: D012 + explicit binary side channel.

Channels:
  0: CT                      (symlink from D012)
  1: norm_dx                 (symlink from D012, already [-1, +1] normalized)
  2: norm_dy                 (symlink)
  3: norm_dz                 (symlink)
  4: side_mask               NEW = (norm_dx > 0).astype(float32)
                             — explicit binary L/R indicator (right of sacrum = 1).

Labels: identical to D012 (7-class border-core).
Motivation (2026-05-26 mode-collapse diagnosis): D012 w40+F-CIRL run had 3/6 folds
collapse LH/RH despite the existing signed normalized offset. Hypothesis: a hard
binary side channel makes the L/R signal harder to ignore, breaking the
mirror-symmetry minimum.

Usage:
    python3 scripts/build_d014_sidemask.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
SRC = NRAW / "Dataset012_PENGWIN_BorderCorePelvic"
DST = NRAW / "Dataset014_PENGWIN_BorderCore_SideMask"

LABELS = {
    "background": 0,
    "sacrum_core": 1, "sacrum_border": 2,
    "leftHip_core": 3, "leftHip_border": 4,
    "rightHip_core": 5, "rightHip_border": 6,
}


def relink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    cids = sorted({f.name[:11] for f in (SRC / "imagesTr").glob("*_0000.nii.gz")})
    if args.limit:
        cids = cids[: args.limit]
    print(f"[D014] {len(cids)} cases from {SRC.name}")

    for i, cid in enumerate(cids):
        for c in range(4):
            relink(SRC / "imagesTr" / f"{cid}_{c:04d}.nii.gz",
                   DST / "imagesTr" / f"{cid}_{c:04d}.nii.gz")
        # channel 4 = side_mask from offset_x (_0001)
        ox_img = sitk.ReadImage(str(SRC / "imagesTr" / f"{cid}_0001.nii.gz"))
        side = (sitk.GetArrayFromImage(ox_img) > 0).astype(np.float32)
        side_img = sitk.GetImageFromArray(side)
        side_img.CopyInformation(ox_img)
        sitk.WriteImage(side_img, str(DST / "imagesTr" / f"{cid}_0004.nii.gz"))
        relink(SRC / "labelsTr" / f"{cid}.nii.gz",
               DST / "labelsTr" / f"{cid}.nii.gz")
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid}")

    ds = {
        "channel_names": {"0": "CT", "1": "offset1", "2": "offset2",
                          "3": "offset3", "4": "side_mask"},
        "labels": LABELS,
        "numTraining": len(cids),
        "file_ending": ".nii.gz",
        "name": "Dataset014_PENGWIN_BorderCore_SideMask",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D014] dataset.json: 5 channels, 7 labels, {len(cids)} cases")
    print(f"[D014] next: nnUNetv2_plan_and_preprocess -d 14 -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
