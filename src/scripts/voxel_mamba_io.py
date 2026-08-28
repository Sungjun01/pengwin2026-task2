"""I/O helpers for standalone full-voxel Mamba experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class LoadedVolume:
    array: np.ndarray
    spacing_zyx: tuple[float, float, float]
    image: sitk.Image


def _spacing_zyx(image: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = image.GetSpacing()
    return (float(sz), float(sy), float(sx))


def load_lps_volume(path: str | Path, dtype: np.dtype | type | None = None) -> LoadedVolume:
    """Read a medical volume, orient to LPS, and return array axes as Z/Y/X."""
    image = sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")
    array = sitk.GetArrayFromImage(image)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return LoadedVolume(array=array, spacing_zyx=_spacing_zyx(image), image=image)


def normalize_hu(
    hu: np.ndarray,
    clip_range: tuple[float, float] = (-1000.0, 2000.0),
) -> np.ndarray:
    """Clip HU values and map the clipped interval linearly to [-1, 1]."""
    lo, hi = float(clip_range[0]), float(clip_range[1])
    if hi <= lo:
        raise ValueError(f"clip_range must be increasing, got {clip_range}")
    clipped = np.clip(hu.astype(np.float32, copy=False), lo, hi)
    return ((clipped - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32, copy=False)


def save_label_volume_like(
    labels_zyx: np.ndarray,
    reference_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save an int label array using a reference image's LPS metadata."""
    reference = sitk.DICOMOrient(sitk.ReadImage(str(reference_path)), "LPS")
    out = sitk.GetImageFromArray(labels_zyx.astype(np.int32, copy=False))
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(output_path))
