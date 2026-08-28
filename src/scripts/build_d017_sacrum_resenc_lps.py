"""Build Dataset017_PENGWIN_SacrumResEnc_LPS — LPS sacrum binary for ResEnc Stage-1.

Replaces Dataset006 (mixed-orientation from D005) with LPS volumes consistent
with D015. Fast path: CT symlinks from D015 channel 0; sacrum mask from D015
7-class labels (core=1 + border=2). Fallback: raw PENGWIN_train + DICOMOrient.

Usage:
    python3 scripts/build_d017_sacrum_resenc_lps.py [--limit N] [--from-raw]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
SRC_D015 = NRAW / "Dataset015_PENGWIN_BorderCore_OrientFixed"
SRC_D012 = NRAW / "Dataset012_PENGWIN_BorderCorePelvic"
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
DST = NRAW / "Dataset017_PENGWIN_SacrumResEnc_LPS"

SACRUM_CORE, SACRUM_BORDER = 1, 2


def case_ids_d015():
    return sorted({f.name[:11] for f in (SRC_D015 / "imagesTr").glob("*_0000.nii.gz")})


def case_ids_d012():
    return sorted({f.name[:11] for f in (SRC_D012 / "imagesTr").glob("*_0000.nii.gz")})


def save_volume(arr_zyx: np.ndarray, ref_img: sitk.Image, path: str, dtype=None):
    if dtype is not None:
        arr_zyx = arr_zyx.astype(dtype)
    out = sitk.GetImageFromArray(arr_zyx)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, path)


def link_ct(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def build_from_d015(cids: list[str]) -> tuple[int, list[str]]:
    built, skipped = 0, []
    for i, cid in enumerate(cids):
        ct_src = SRC_D015 / "imagesTr" / f"{cid}_0000.nii.gz"
        lab_src = SRC_D015 / "labelsTr" / f"{cid}.nii.gz"
        if not ct_src.exists() or not lab_src.exists():
            skipped.append(f"{cid}:missing-d015")
            continue
        lab = sitk.ReadImage(str(lab_src))
        labarr = sitk.GetArrayFromImage(lab)
        sac = ((labarr == SACRUM_CORE) | (labarr == SACRUM_BORDER)).astype(np.uint8)
        if sac.sum() < 1:
            skipped.append(f"{cid}:no-sacrum")
            continue
        link_ct(ct_src, DST / "imagesTr" / f"{cid}_0000.nii.gz")
        save_volume(sac, lab, str(DST / "labelsTr" / f"{cid}.nii.gz"), dtype=np.uint8)
        built += 1
        if (i + 1) % 40 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid}")
    return built, skipped


def build_from_raw(cids: list[str]) -> tuple[int, list[str]]:
    built, skipped = 0, []
    for i, cid in enumerate(cids):
        num = cid.replace("PENGWIN_", "")
        src_img = RAW / num / "image.mha"
        src_lab = RAW / num / "label.mha"
        if not src_img.exists() or not src_lab.exists():
            skipped.append(f"{cid}:missing")
            continue
        img = sitk.DICOMOrient(sitk.ReadImage(str(src_img)), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(str(src_lab)), "LPS")
        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        labarr = sitk.GetArrayFromImage(lab)
        sac = ((labarr >= 1) & (labarr <= 50)).astype(np.uint8)
        if sac.sum() < 1:
            skipped.append(f"{cid}:no-sacrum")
            continue
        save_volume(ct, img, str(DST / "imagesTr" / f"{cid}_0000.nii.gz"))
        save_volume(sac, img, str(DST / "labelsTr" / f"{cid}.nii.gz"), dtype=np.uint8)
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid}")
    return built, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-raw", action="store_true", help="reorient from PENGWIN_train mha")
    args = ap.parse_args()

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    if args.from_raw or not SRC_D015.exists():
        cids = case_ids_d012()
        mode = "raw+LPS"
    else:
        cids = case_ids_d015()
        mode = "D015-symlink"
    if args.limit:
        cids = cids[: args.limit]
    print(f"[D017] {len(cids)} cases mode={mode}")

    if mode == "D015-symlink":
        built, skipped = build_from_d015(cids)
    else:
        built, skipped = build_from_raw(cids)

    ds = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "sacrum": 1},
        "numTraining": built,
        "file_ending": ".nii.gz",
        "name": "Dataset017_PENGWIN_SacrumResEnc_LPS",
        "description": (
            "PENGWIN pelvic sacrum-only binary, LPS-oriented (D015 frame). "
            "ResEnc-M Stage-1 replacement for Dataset006. 170 pelvic cases."
        ),
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D017] built {built} cases, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped[:10]}...")
    print(
        "  next: nnUNetv2_plan_and_preprocess -d 17 -pl nnUNetPlannerResEncM "
        "-c 3d_fullres --verify_dataset_integrity"
    )


if __name__ == "__main__":
    main()
