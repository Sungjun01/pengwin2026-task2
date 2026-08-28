"""I/O helpers for SimpleITK-format label volumes (.nii.gz / .mha)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_label_volume(
    path: Path | str,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a label image and return (array, spacing_zyx).

    Spacing is returned in array-axis order (z, y, x) for direct use with
    numpy-indexed masks. The underlying SimpleITK convention is (x, y, z).
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.int32)
    sx, sy, sz = img.GetSpacing()
    return arr, (sz, sy, sx)


def list_case_files(
    directory: Path | str,
    extensions: tuple[str, ...] = (".nii.gz", ".mha"),
) -> list[Path]:
    """Return sorted list of label files in a directory."""
    d = Path(directory)
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and any(p.name.endswith(ext) for ext in extensions)
    )


def pair_predictions_with_gt(
    pred_dir: Path | str,
    gt_dir: Path | str,
) -> list[tuple[Path, Path]]:
    """Pair files in pred_dir with files in gt_dir by basename."""
    pred_files = list_case_files(pred_dir)
    gt_files = {p.name: p for p in list_case_files(gt_dir)}
    pairs = []
    for p in pred_files:
        if p.name in gt_files:
            pairs.append((p, gt_files[p.name]))
    return pairs
