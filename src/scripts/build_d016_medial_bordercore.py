"""Build Dataset016_PENGWIN_BorderCore_MedialOrient: D015 + medial-axis adaptive border.

Identical to D015 in every channel (CT + 3 offsets + side_mask) and in the LPS
canonicalization. The single difference is the label: instead of fixed seam
half-width t=2 voxels, the seam half-width is per-fragment medial-axis adaptive
(ceil(half_thickness / 3), clipped to [1, 3]). Motivation: the 1st-place PENGWIN
2024 method (MIC-DKFZ ABBC) uses exactly this dynamic boundary thickness to keep
thin fragments from being eroded into core-less voxels — the failure mode that
shows up as ``sacrum 1 → 2 over-seg'' in our D015 fold_0 ep 50 measurement.

To save disk and build time, the 5-channel imagesTr files are *symlinked* from
the existing D015 build (they are identical between D015 and D016). Only the
labelsTr/ files are rebuilt.

Usage:
    python3 scripts/build_d016_medial_bordercore.py [--limit N]
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
from scripts.border_core_conversion import PELVIC_ANATOMY
from scripts.border_core_conversion_medial import convert_to_border_core_medial

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
SRC_D015 = NRAW / "Dataset015_PENGWIN_BorderCore_OrientFixed"
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
DST = NRAW / "Dataset016_PENGWIN_BorderCore_MedialOrient"

LABELS = {
    "background": 0,
    "sacrum_core": 1, "sacrum_border": 2,
    "leftHip_core": 3, "leftHip_border": 4,
    "rightHip_core": 5, "rightHip_border": 6,
}

# Adaptive-t hyperparameters (see border_core_conversion_medial module docstring).
# Defaults match the unit-test expectations. Override per-case via env if tuning.
DIVISOR = float(os.environ.get("D016_DIVISOR", "3.0"))
T_MIN = int(os.environ.get("D016_T_MIN", "1"))
T_MAX = int(os.environ.get("D016_T_MAX", "3"))


def case_ids_from_d015():
    """Use D015's already-built case list to guarantee channel↔label parity."""
    return sorted({f.name[:11] for f in (SRC_D015 / "imagesTr").glob("*_0000.nii.gz")})


def symlink_channels(cid: str):
    """Symlink the 5 imagesTr channels from D015 to D016 (idempotent)."""
    for ch in ("0000", "0001", "0002", "0003", "0004"):
        src = SRC_D015 / "imagesTr" / f"{cid}_{ch}.nii.gz"
        dst = DST / "imagesTr" / f"{cid}_{ch}.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def save_label(arr_zyx: np.ndarray, ref_img: sitk.Image, path: str):
    out = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="build only first N cases (for sanity)")
    args = ap.parse_args()

    if not SRC_D015.exists():
        raise SystemExit(f"D015 not found at {SRC_D015}. Build D015 first.")

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    cids = case_ids_from_d015()
    if args.limit:
        cids = cids[: args.limit]
    print(f"[D016] {len(cids)} cases (medial-axis adaptive border; "
          f"divisor={DIVISOR} t∈[{T_MIN},{T_MAX}])")

    built, skipped = 0, []
    for i, cid in enumerate(cids):
        num = cid.replace("PENGWIN_", "")
        src_lab = RAW / num / "label.mha"
        if not src_lab.exists():
            skipped.append(f"{cid}:missing-label"); continue

        # Reorient label to LPS — matches D015 exactly so channels↔label aligned.
        lab = sitk.DICOMOrient(sitk.ReadImage(str(src_lab)), "LPS")
        labarr = sitk.GetArrayFromImage(lab)

        sac_mask = ((labarr >= 1) & (labarr <= 50)).astype(np.float32)
        if sac_mask.sum() < 1e-3:
            skipped.append(f"{cid}:no-sacrum"); continue

        # Medial-axis adaptive seam — the only line that differs from D015.
        bc = convert_to_border_core_medial(
            labarr, PELVIC_ANATOMY,
            divisor=DIVISOR, t_min=T_MIN, t_max=T_MAX,
        )

        symlink_channels(cid)
        save_label(bc, lab, str(DST / "labelsTr" / f"{cid}.nii.gz"))
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid}")

    ds = {
        "channel_names": {"0": "CT", "1": "offset1", "2": "offset2",
                          "3": "offset3", "4": "side_mask"},
        "labels": LABELS,
        "numTraining": built,
        "file_ending": ".nii.gz",
        "name": "Dataset016_PENGWIN_BorderCore_MedialOrient",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D016] built {built} cases, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped detail: {skipped[:10]}...")
    print(f"  next: nnUNetv2_plan_and_preprocess -d 16 -c 3d_fullres --verify_dataset_integrity -np 8")


if __name__ == "__main__":
    main()
