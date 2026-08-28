import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from evaluation.io_utils import load_label_volume


def _write_temp_label(arr: np.ndarray, spacing: tuple[float, float, float]) -> Path:
    img = sitk.GetImageFromArray(arr.astype(np.int32))
    img.SetSpacing(spacing[::-1])  # SITK uses (x, y, z); array is (z, y, x)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    f.close()
    sitk.WriteImage(img, f.name)
    return Path(f.name)


def test_load_label_volume_roundtrip():
    arr = np.zeros((5, 6, 7), dtype=np.int32)
    arr[1:3, 2:4, 3:5] = 7
    path = _write_temp_label(arr, spacing=(1.0, 2.0, 3.0))
    try:
        lbl, spacing = load_label_volume(path)
        assert lbl.shape == arr.shape
        np.testing.assert_array_equal(lbl, arr)
        assert spacing == pytest.approx((1.0, 2.0, 3.0))
    finally:
        path.unlink()


def test_load_label_volume_supports_mha():
    arr = np.zeros((4, 4, 4), dtype=np.int32)
    arr[1, 1, 1] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    f = tempfile.NamedTemporaryFile(suffix=".mha", delete=False)
    f.close()
    sitk.WriteImage(img, f.name)
    try:
        lbl, spacing = load_label_volume(Path(f.name))
        assert lbl.shape == (4, 4, 4)
        assert spacing == pytest.approx((0.5, 0.5, 0.5))
    finally:
        Path(f.name).unlink()
