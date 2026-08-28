"""End-to-end smoke test: 3 methods × 3 cases through full pipeline."""
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluation import evaluate_submission, rank_submissions


def _write_lbl(arr, path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr.astype(np.int32))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, str(path))


def _build_realistic_case(shape=(40, 40, 60)):
    """One CT-like case with 2 sacrum + 1 left hip + 1 right hip + 1 femur."""
    lbl = np.zeros(shape, dtype=np.int32)
    lbl[5:10, 5:10, 5:10] = 1   # sacrum frag 1
    lbl[5:10, 5:10, 15:20] = 2  # sacrum frag 2
    lbl[5:10, 15:20, 5:10] = 51   # leftHip
    lbl[5:10, 25:30, 5:10] = 101  # rightHip
    lbl[20:30, 15:25, 30:40] = 151  # femur
    return lbl


def test_three_methods_full_pipeline():
    """3 methods (perfect / small_error / missing) × 3 cases through ranking."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dirs = {n: td / n for n in ["gt", "perfect", "small_error", "missing"]}
        for d in dirs.values():
            d.mkdir()

        for i in range(3):
            gt = _build_realistic_case()
            _write_lbl(gt, dirs["gt"] / f"case_{i:03d}.nii.gz")
            _write_lbl(gt.copy(), dirs["perfect"] / f"case_{i:03d}.nii.gz")

            shifted = np.zeros_like(gt)
            shifted[6:11, 5:10, 5:10] = 1     # sacrum1 shifted 1 voxel
            shifted[5:10, 5:10, 15:20] = 2
            shifted[5:10, 15:20, 5:10] = 51
            shifted[5:10, 25:30, 5:10] = 101
            shifted[20:30, 15:25, 30:40] = 151
            _write_lbl(shifted, dirs["small_error"] / f"case_{i:03d}.nii.gz")

            partial = gt.copy()
            partial[partial == 2] = 0  # drop sacrum frag 2
            _write_lbl(partial, dirs["missing"] / f"case_{i:03d}.nii.gz")

        result = rank_submissions(
            submissions={
                "perfect": dirs["perfect"],
                "small_error": dirs["small_error"],
                "missing": dirs["missing"],
            },
            gt_dir=dirs["gt"],
            output_path=td / "ranking.json",
        )

        assert result["ordering"][0] == "perfect"
        assert result["ordering"][-1] == "missing"
        assert result["per_metric_ranks"]["IoU_F"]["perfect"] == 1
        assert (td / "ranking.json").exists()
