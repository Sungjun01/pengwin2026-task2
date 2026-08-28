import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from evaluation.ranking import rank_dict, rank_submissions
from evaluation.tests.fixtures.synthetic_cases import (
    perfect_prediction, case_with_missing_fragment,
)


def _write_lbl(arr, path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr.astype(np.int32))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, str(path))


# --- rank_dict ---

def test_rank_dict_higher_better():
    scores = {"a": 0.9, "b": 0.7, "c": 0.8}
    ranks = rank_dict(scores, higher_better=True)
    assert ranks == {"a": 1, "c": 2, "b": 3}


def test_rank_dict_lower_better():
    scores = {"a": 1.0, "b": 0.5, "c": 0.8}
    ranks = rank_dict(scores, higher_better=False)
    assert ranks == {"b": 1, "c": 2, "a": 3}


def test_rank_dict_tie_uses_min_rank():
    scores = {"a": 0.9, "b": 0.9, "c": 0.5}
    ranks = rank_dict(scores, higher_better=True)
    assert ranks == {"a": 1, "b": 1, "c": 3}


# --- rank_submissions ---

def test_rank_submissions_perfect_beats_imperfect():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gt_dir = td / "gt"
        good_dir = td / "good"
        bad_dir = td / "bad"
        gt_dir.mkdir(); good_dir.mkdir(); bad_dir.mkdir()

        p, g, _ = perfect_prediction(n=3)
        _write_lbl(g, gt_dir / "case_001.nii.gz")
        _write_lbl(p, good_dir / "case_001.nii.gz")
        bad_pred, _, _ = case_with_missing_fragment(n_gt=3, n_pred=2)
        _write_lbl(bad_pred, bad_dir / "case_001.nii.gz")

        result = rank_submissions(
            submissions={"good": good_dir, "bad": bad_dir},
            gt_dir=gt_dir,
        )
        assert result["ordering"] == ["good", "bad"]
        assert result["final_rank"]["good"] < result["final_rank"]["bad"]


def test_rank_submissions_tie_broken_by_time():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gt_dir = td / "gt"
        a_dir = td / "a"
        b_dir = td / "b"
        gt_dir.mkdir(); a_dir.mkdir(); b_dir.mkdir()

        p, g, _ = perfect_prediction(n=2)
        _write_lbl(g, gt_dir / "case_001.nii.gz")
        _write_lbl(p, a_dir / "case_001.nii.gz")
        _write_lbl(p, b_dir / "case_001.nii.gz")

        result = rank_submissions(
            submissions={"slow": a_dir, "fast": b_dir},
            gt_dir=gt_dir,
            execution_times={"slow": 300.0, "fast": 60.0},
        )
        assert result["ordering"][0] == "fast"


def test_rank_submissions_writes_json(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir(); pred_dir.mkdir()
    p, g, _ = perfect_prediction(n=2)
    _write_lbl(g, gt_dir / "case_001.nii.gz")
    _write_lbl(p, pred_dir / "case_001.nii.gz")
    out = tmp_path / "ranking.json"
    rank_submissions(
        submissions={"only": pred_dir},
        gt_dir=gt_dir,
        output_path=out,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert "final_rank" in data and "only" in data["final_rank"]


def test_ranking_cli_smoke(tmp_path):
    gt_dir = tmp_path / "gt"
    a_dir = tmp_path / "a"
    gt_dir.mkdir(); a_dir.mkdir()
    p, g, _ = perfect_prediction(n=2)
    _write_lbl(g, gt_dir / "case_001.nii.gz")
    _write_lbl(p, a_dir / "case_001.nii.gz")

    out = tmp_path / "ranking.json"
    ret = subprocess.run(
        [sys.executable, "-m", "evaluation.ranking",
         "-s", f"only:{a_dir}", "-g", str(gt_dir), "-o", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).parents[2],
    )
    assert ret.returncode == 0, ret.stderr
    assert out.exists()
