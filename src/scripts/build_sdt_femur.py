"""Dataset023 = SDT (Skeleton-aware Distance Transform) femur target — MICCAI'23, arXiv:2310.05262.

Per femur fragment (raw instance labels 151-200), build the SDT energy
    E(x) = ( d(x, boundary) / ( d(x, skeleton) + d(x, boundary) ) )^alpha ,  alpha=0.8
boundary -> 0, skeleton -> 1, exactly ONE ridge per fragment. Fracture seams between two
fragments are each fragment's boundary -> energy 0 there -> low-energy VALLEYS that separate the
per-fragment skeleton ridges. Quantize E into 10 bins (+ background) = an 11-class nnUNet target
(the paper's optimal "classification mode"); training is then plain nnUNet (no custom loss).

Decode (separate, scripts later): predicted energy>0.7 -> skeleton -> connected components = seeds
-> watershed on reversed energy -> femur fragment instances. (Replaces the geometric watershed that
splits at the wrong place; this splits at the LEARNED fracture surface.)

CT (1ch) is symlinked from Dataset004. Mirrors scripts/build_d021_fsdf_femur.py (loading + LPS orient).
Usage: python -m scripts.build_sdt_femur [--limit N] [--sanity]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label as cc_label
from skimage.morphology import skeletonize

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SRC04 = NRAW / "Dataset004_PENGWIN_FemurAnatomy"
DST = NRAW / "Dataset023_PENGWIN_SDT_Femur"
LO, HI = 151, 200
ALPHA = 0.8
NBINS = 10
PAD = 4


def _bbox_slice(mask, pad, shape):
    zz, yy, xx = np.where(mask)
    return tuple(
        slice(max(0, a.min() - pad), min(n, a.max() + pad + 1))
        for a, n in zip((zz, yy, xx), shape)
    )


def sdt_energy(inst, spacing_xyz):
    """Return (E float32 in [0,1] on femur voxels, n_frag, n_skel_cc) for the whole volume."""
    samp = np.array(spacing_xyz[::-1], dtype=np.float32)  # (z,y,x)
    E = np.zeros(inst.shape, np.float32)
    frags = [int(f) for f in np.unique(inst) if LO <= f <= HI]
    n_skel_cc = 0
    for f in frags:
        m_full = inst == f
        if m_full.sum() < 10:
            continue
        sl = _bbox_slice(m_full, PAD, inst.shape)
        m = m_full[sl]
        d_b = distance_transform_edt(m, sampling=samp).astype(np.float32)   # dist to boundary
        sk = skeletonize(m)
        if not sk.any():
            sk = m & (d_b >= d_b.max() - 1e-3)                              # degenerate fallback
        n_skel_cc += cc_label(sk)[1]
        d_s = distance_transform_edt(~sk, sampling=samp).astype(np.float32)  # dist to skeleton
        denom = d_s + d_b
        e = np.zeros(m.shape, np.float32)
        v = m & (denom > 0)
        e[v] = (d_b[v] / denom[v]) ** ALPHA
        e[sk] = 1.0
        sub = E[sl]
        sub[m] = e[m]
    return E, len(frags), n_skel_cc


def energy_to_bins(E, fem_mask):
    out = np.zeros(E.shape, np.uint8)
    b = np.clip(np.floor(E[fem_mask] * NBINS).astype(int), 0, NBINS - 1) + 1  # 1..NBINS
    out[fem_mask] = b
    return out


def _load_inst(num):
    gimg = sitk.DICOMOrient(sitk.ReadImage(str(RAW / num / "label.mha")), "LPS")
    gt = sitk.GetArrayFromImage(gimg).astype(int)
    gt = np.where((gt >= LO) & (gt <= HI), gt, 0)
    return gt, gimg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    cases = sorted(p.name.replace(".nii.gz", "") for p in (SRC04 / "labelsTr").glob("PENGWIN_*.nii.gz"))
    if args.sanity:
        cases = cases[:2]
    elif args.limit:
        cases = cases[: args.limit]

    if args.sanity:
        for cid in cases:
            num = cid.replace("PENGWIN_", "")
            gt, gimg = _load_inst(num)
            E, nfrag, nskel = sdt_energy(gt, gimg.GetSpacing())
            fem = (gt >= LO) & (gt <= HI)
            bins = energy_to_bins(E, fem)
            hist = np.bincount(bins[fem], minlength=NBINS + 1)
            print(f"[sanity] {cid}: frags={nfrag} skel_CCs={nskel} "
                  f"E[min/max]={E[fem].min():.3f}/{E[fem].max():.3f} "
                  f"bin_hist(1..10)={hist[1:].tolist()}")
        print("SANITY_OK (skel_CCs≈frags 이면 조각별 능선 분리 정상)")
        return

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    print(f"[SDT] {len(cases)} femur cases -> {DST}", flush=True)
    built = 0
    for i, cid in enumerate(cases):
        num = cid.replace("PENGWIN_", "")
        if not (RAW / num / "label.mha").exists():
            print(f"  skip {cid}: no raw label"); continue
        gt, gimg = _load_inst(num)
        E, _, _ = sdt_energy(gt, gimg.GetSpacing())
        fem = (gt >= LO) & (gt <= HI)
        lab = energy_to_bins(E, fem)
        lab_img = sitk.GetImageFromArray(lab); lab_img.CopyInformation(gimg)
        sitk.WriteImage(lab_img, str(DST / "labelsTr" / f"{cid}.nii.gz"))
        src = SRC04 / "imagesTr" / f"{cid}_0000.nii.gz"
        dst = DST / "imagesTr" / f"{cid}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(cases)}] {cid}", flush=True)

    labels = {"background": 0, **{f"sdt_bin{i}": i for i in range(1, NBINS + 1)}}
    ds = {"channel_names": {"0": "CT"}, "labels": labels, "numTraining": built,
          "file_ending": ".nii.gz", "name": "Dataset023_PENGWIN_SDT_Femur"}
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[SDT] built={built} -> {DST}\nlabels={labels}\nSDT_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
