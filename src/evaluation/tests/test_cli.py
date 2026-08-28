import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluation.tests.fixtures.synthetic_cases import perfect_prediction


def _write_lbl(arr, path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr.astype(np.int32))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, str(path))


def test_pengwin_eval_cli_smoke():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred_dir = td / "pred"
        gt_dir = td / "gt"
        pred_dir.mkdir(); gt_dir.mkdir()
        p, g, _ = perfect_prediction(n=2)
        _write_lbl(p, pred_dir / "case_001.nii.gz")
        _write_lbl(g, gt_dir / "case_001.nii.gz")
        out = td / "result.json"

        ret = subprocess.run(
            [sys.executable, "-m", "evaluation.pengwin_eval",
             "-p", str(pred_dir), "-g", str(gt_dir), "-o", str(out)],
            capture_output=True, text=True,
            cwd=Path(__file__).parents[2],
        )
        assert ret.returncode == 0, ret.stderr
        assert out.exists()
        with open(out) as f:
            result = json.load(f)
        assert result["aggregate"]["IoU_F_mean"] == 1.0
