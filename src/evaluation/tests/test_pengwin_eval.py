import numpy as np
import pytest

import json
import tempfile
from pathlib import Path

import SimpleITK as sitk

from evaluation.pengwin_eval import (
    evaluate_f_level, evaluate_a_level, evaluate_case, evaluate_submission,
)
from evaluation.tests.fixtures.synthetic_cases import (
    perfect_prediction, case_with_missing_fragment, n_disjoint_fragments,
)


def _write_lbl(arr: np.ndarray, path: Path, spacing=(1.0, 1.0, 1.0)) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.int32))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, str(path))


def test_f_level_perfect_prediction():
    pred, gt, spacing = perfect_prediction(n=3)
    r = evaluate_f_level(gt, pred, spacing)
    assert r["IoU_F"] == pytest.approx(1.0)
    assert r["HD95_F_mm"] == pytest.approx(0.0)
    assert r["ASSD_F_mm"] == pytest.approx(0.0)


def test_f_level_missing_fragment_penalty():
    pred, gt, spacing = case_with_missing_fragment(n_gt=3, n_pred=2)
    r = evaluate_f_level(gt, pred, spacing)
    # Two perfect (IoU=1) + one missing (IoU=0): mean = 2/3
    assert r["IoU_F"] == pytest.approx(2.0 / 3.0)
    # Bounding sphere diameter of a 27-voxel cube at unit spacing
    expected_d = 2 * (3 * 27 / (4 * np.pi)) ** (1 / 3)
    expected_hd95_mean = (0 + 0 + expected_d) / 3
    assert r["HD95_F_mm"] == pytest.approx(expected_hd95_mean, rel=1e-5)


def test_f_level_no_gt_fragments_returns_nans():
    pred = np.zeros((10, 10, 10), dtype=np.int32)
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    r = evaluate_f_level(gt, pred, (1.0, 1.0, 1.0))
    assert np.isnan(r["IoU_F"])
    assert np.isnan(r["HD95_F_mm"])
    assert np.isnan(r["ASSD_F_mm"])


def test_a_level_perfect_prediction():
    gt, spacing = n_disjoint_fragments(n=2, anatomy_offset=0)
    r = evaluate_a_level(gt, gt.copy(), spacing)
    assert r["IoU_A"] == pytest.approx(1.0)
    assert r["HD95_A_mm"] == pytest.approx(0.0)


def test_a_level_merged_fragments_anatomy_correct():
    """Fragment-level wrong (one merged blob) but anatomy-level correct."""
    gt, spacing = n_disjoint_fragments(n=2, anatomy_offset=0)
    pred = gt.copy()
    pred[pred == 2] = 1  # merge two sacrum fragments into one
    r = evaluate_a_level(gt, pred, spacing)
    assert r["IoU_A"] == pytest.approx(1.0)


def test_a_level_skips_absent_bones():
    gt, spacing = n_disjoint_fragments(n=2, anatomy_offset=0)
    r = evaluate_a_level(gt, gt.copy(), spacing)
    # Only sacrum present; absent bones (LH/RH/femur) shouldn't pull the mean down
    assert r["IoU_A"] == pytest.approx(1.0)


def test_evaluate_case_returns_six_metrics():
    pred, gt, _ = perfect_prediction(n=2)
    with tempfile.TemporaryDirectory() as td:
        pp = Path(td) / "pred.nii.gz"
        gp = Path(td) / "gt.nii.gz"
        _write_lbl(pred, pp)
        _write_lbl(gt, gp)
        r = evaluate_case(pp, gp)
    for key in ["IoU_F", "HD95_F_mm", "ASSD_F_mm",
                "IoU_A", "HD95_A_mm", "ASSD_A_mm"]:
        assert key in r
        assert not np.isnan(r[key])


def test_evaluate_submission_aggregates_multiple_cases():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred_dir = td / "pred"
        gt_dir = td / "gt"
        pred_dir.mkdir(); gt_dir.mkdir()

        for i in range(3):
            p, g, _ = perfect_prediction(n=2 + i)
            _write_lbl(p, pred_dir / f"case_{i:03d}.nii.gz")
            _write_lbl(g, gt_dir / f"case_{i:03d}.nii.gz")

        out_json = td / "metrics.json"
        result = evaluate_submission(pred_dir, gt_dir, output_path=out_json)

        assert len(result["per_case"]) == 3
        assert result["aggregate"]["IoU_F_mean"] == pytest.approx(1.0)
        assert out_json.exists()
        with open(out_json) as f:
            loaded = json.load(f)
        assert "schema_version" in loaded
        assert "per_case" in loaded and "aggregate" in loaded


def test_evaluate_submission_handles_missing_pred_case():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred_dir = td / "pred"
        gt_dir = td / "gt"
        pred_dir.mkdir(); gt_dir.mkdir()
        p, g, _ = perfect_prediction(n=2)
        _write_lbl(g, gt_dir / "case_001.nii.gz")
        _write_lbl(g, gt_dir / "case_002.nii.gz")
        _write_lbl(p, pred_dir / "case_001.nii.gz")
        result = evaluate_submission(pred_dir, gt_dir)
    assert len(result["per_case"]) == 1
