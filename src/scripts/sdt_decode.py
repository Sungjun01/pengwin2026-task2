"""SDT decode (arXiv:2310.05262): a predicted 11-class energy-bin map -> femur fragment
instances. Validated property: energy>0.7 gives exactly ONE 26-connected seed per fragment.

  energy = (bin - 0.5)/10  (bg=0)
  seeds  = 26-conn CC of (energy > 0.7), tiny seeds dropped
  inst   = watershed on REVERSED energy within femur foreground (skeleton basins grow to
           boundaries; fracture seams = where two basins meet = the split)

This REPLACES the geometric distance-transform watershed (which splits at the wrong place);
SDT splits at the LEARNED fracture surface. Train-side is plain nnUNet (11-class); only this
decode is custom. Inference cost = one CC + one watershed per femur (must pass T4 time gate).

Usage (gate): python -m scripts.sdt_decode <sem_pred_dir> <out_instance_dir>
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, label as cc
from skimage.segmentation import watershed

S26 = generate_binary_structure(3, 3)
NBINS = 10
SEED_THR = 0.7
MIN_SEED = 50          # voxels; drop spurious sub-skeleton seeds
FEMUR_BASE = 151       # output instance ids start here (PENGWIN femur range 151-200)


def bins_to_energy(sem):
    e = np.zeros(sem.shape, np.float32)
    fem = sem >= 1
    e[fem] = (sem[fem].astype(np.float32) - 0.5) / NBINS   # bin center -> [0.05, 0.95]
    return e, fem


def sdt_instances(sem, seed_thr=SEED_THR, min_seed=MIN_SEED, femur_base=FEMUR_BASE):
    """sem: (Z,Y,X) int 0..10. Returns uint16 instance map (ids femur_base, +1, ...)."""
    sem = np.asarray(sem)
    energy, fem = bins_to_energy(sem)
    if not fem.any():
        return np.zeros(sem.shape, np.uint16)
    seed = energy > seed_thr
    lab, n = cc(seed, S26)
    if n == 0:
        out = np.zeros(sem.shape, np.uint16)
        out[fem] = femur_base
        return out
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(lab.shape, bool)
    for i in range(1, n + 1):
        if sizes[i] >= min_seed:
            keep |= lab == i
    markers, nm = cc(keep, S26)
    if nm == 0:
        out = np.zeros(sem.shape, np.uint16)
        out[fem] = femur_base
        return out
    ws = watershed(-energy, markers=markers, mask=fem)     # basins grow from skeleton to boundary
    out = np.zeros(sem.shape, np.uint16)
    fg = ws > 0
    out[fg] = ws[fg].astype(np.uint16) + (femur_base - 1)   # ids -> 151,152,...
    return out


def main():
    sem_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{sem_dir}/PENGWIN_*.nii.gz")):
        im = sitk.ReadImage(f)
        sem = sitk.GetArrayFromImage(im).astype(np.int32)
        inst = sdt_instances(sem)
        o = sitk.GetImageFromArray(inst); o.CopyInformation(im)
        sitk.WriteImage(o, f"{out_dir}/{os.path.basename(f)}")
        n += 1
    print(f"SDT-decoded {n} -> {out_dir}")


if __name__ == "__main__":
    main()
