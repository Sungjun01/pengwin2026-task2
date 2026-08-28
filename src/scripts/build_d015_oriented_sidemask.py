"""Build Dataset015_PENGWIN_BorderCore_OrientFixed: orientation-canonicalized D014.

Bug fix (2026-05-26): raw PENGWIN images are stored in mixed orientations (LPS and
RAS-style flipped axially). The norm_dx in D010/D012/D014 was computed in image
voxel coords, so its sign maps to patient L/R *inconsistently* across cases —
about 5/8 sampled cases have LH on +voxel-x side, the rest have it on -voxel-x.
This silently provided conflicting LR signal and best explains the fold-dependent
mode collapse observed across w139 and w40+F-CIRL runs.

D015 reorients all CT + GT to LPS first (sitk.DICOMOrient), then recomputes
offset channels and side_mask in the canonical frame. Result: norm_dx > 0 ↔
patient's L ↔ leftHip side, consistently across every case.

Channels (5):
  0  CT
  1  norm_dx   (LPS-aligned, +x = patient L)
  2  norm_dy
  3  norm_dz
  4  side_mask = (norm_dx > 0)   # 1 = LH side, 0 = RH side, consistent

Labels: 7-class border-core (same as D012/D014).

Sacrum centroid: from GT label (Oracle), matching the §7 5-way ablation finding
that Predicted ≈ Oracle (<0.001 Dice diff). A predicted-sacrum variant can be
built later for the final inference pipeline if needed.

Usage:
    python3 scripts/build_d015_oriented_sidemask.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.border_core_conversion import convert_to_border_core, PELVIC_ANATOMY

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
SRC_D012 = NRAW / "Dataset012_PENGWIN_BorderCorePelvic"
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
DST = NRAW / "Dataset015_PENGWIN_BorderCore_OrientFixed"

LABELS = {
    "background": 0,
    "sacrum_core": 1, "sacrum_border": 2,
    "leftHip_core": 3, "leftHip_border": 4,
    "rightHip_core": 5, "rightHip_border": 6,
}


def case_ids():
    """Use D012's case list (170 pelvic cases)."""
    return sorted({f.name[:11] for f in (SRC_D012 / "imagesTr").glob("*_0000.nii.gz")})


def compute_offsets(sacrum_mask: np.ndarray, spacing_xyz):
    """sacrum_mask shape (Z, Y, X). spacing_xyz = sitk (sx, sy, sz).

    Returns (norm_dx, norm_dy, norm_dz) each shape (Z, Y, X), clipped to [-1, +1].
    Normalized by half of the case's longest physical extent (matching D010 style).
    """
    cz, cy, cx = center_of_mass(sacrum_mask)
    nz, ny, nx = sacrum_mask.shape
    sx, sy, sz = spacing_xyz
    zc = (np.arange(nz, dtype=np.float32) - np.float32(cz)) * np.float32(sz)
    yc = (np.arange(ny, dtype=np.float32) - np.float32(cy)) * np.float32(sy)
    xc = (np.arange(nx, dtype=np.float32) - np.float32(cx)) * np.float32(sx)
    extents = (nz * sz, ny * sy, nx * sx)
    norm = max(extents) / 2.0
    zc = np.clip(zc / norm, -1.0, 1.0)
    yc = np.clip(yc / norm, -1.0, 1.0)
    xc = np.clip(xc / norm, -1.0, 1.0)
    # broadcast to (Z,Y,X)
    Zg = zc[:, None, None] * np.ones((1, ny, nx), dtype=np.float32)
    Yg = np.ones((nz, 1, nx), dtype=np.float32) * yc[None, :, None]
    Xg = np.ones((nz, ny, 1), dtype=np.float32) * xc[None, None, :]
    return Xg, Yg, Zg


def save_volume(arr_zyx: np.ndarray, ref_img: sitk.Image, path: str, dtype=None):
    if dtype is not None:
        arr_zyx = arr_zyx.astype(dtype)
    out = sitk.GetImageFromArray(arr_zyx)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="build only first N cases (for sanity)")
    args = ap.parse_args()

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    cids = case_ids()
    if args.limit:
        cids = cids[: args.limit]
    print(f"[D015] {len(cids)} cases (reoriented to LPS)")

    built, skipped = 0, []
    for i, cid in enumerate(cids):
        num = cid.replace("PENGWIN_", "")
        src_img = RAW / num / "image.mha"
        src_lab = RAW / num / "label.mha"
        if not src_img.exists() or not src_lab.exists():
            skipped.append(f"{cid}:missing"); continue

        img = sitk.DICOMOrient(sitk.ReadImage(str(src_img)), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(str(src_lab)), "LPS")

        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        labarr = sitk.GetArrayFromImage(lab)
        spacing = img.GetSpacing()

        sac_mask = ((labarr >= 1) & (labarr <= 50)).astype(np.float32)
        if sac_mask.sum() < 1e-3:
            skipped.append(f"{cid}:no-sacrum"); continue

        nx, ny, nz = compute_offsets(sac_mask, spacing)
        side = (nx > 0).astype(np.float32)
        bc = convert_to_border_core(labarr, PELVIC_ANATOMY, t=2)

        save_volume(ct, img, str(DST / "imagesTr" / f"{cid}_0000.nii.gz"))
        save_volume(nx, img, str(DST / "imagesTr" / f"{cid}_0001.nii.gz"))
        save_volume(ny, img, str(DST / "imagesTr" / f"{cid}_0002.nii.gz"))
        save_volume(nz, img, str(DST / "imagesTr" / f"{cid}_0003.nii.gz"))
        save_volume(side, img, str(DST / "imagesTr" / f"{cid}_0004.nii.gz"))
        save_volume(bc, img, str(DST / "labelsTr" / f"{cid}.nii.gz"), dtype=np.uint8)
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid}")

    ds = {
        "channel_names": {"0": "CT", "1": "offset1", "2": "offset2",
                          "3": "offset3", "4": "side_mask"},
        "labels": LABELS,
        "numTraining": built,
        "file_ending": ".nii.gz",
        "name": "Dataset015_PENGWIN_BorderCore_OrientFixed",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D015] built {built} cases, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped detail: {skipped[:10]}...")
    print(f"  next: nnUNetv2_plan_and_preprocess -d 15 -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
