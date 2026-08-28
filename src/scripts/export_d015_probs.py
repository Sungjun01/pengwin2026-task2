"""Export D015 softmax probabilities (.npz) for Form-Learn edge features (§4A.8.2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.inference_v7_d015 import (
    compute_d015_offset_channels,
    init_predictor,
    predict_softmax,
    save_temp_channel,
)

DEFAULT_RAW = Path(
    "/home/schoaq/taehee/nnUNet_data/raw/Dataset015_PENGWIN_BorderCore_OrientFixed"
)
DEFAULT_RESULTS = Path(
    "/home/schoaq/taehee/nnUNet_data/results/Dataset015_PENGWIN_BorderCore_OrientFixed"
    "/nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres"
)
HELDOUT = ("003", "011", "017", "018", "025", "034")


def export_case(case_id: str, images_dir: Path, out_dir: Path, results_dir: Path) -> None:
    import tempfile

    ct_path = images_dir / f"PENGWIN_{case_id}_0000.nii.gz"
    if not ct_path.exists():
        raise FileNotFoundError(ct_path)
    ct_lps = sitk.DICOMOrient(sitk.ReadImage(str(ct_path)), "LPS")
    spacing = ct_lps.GetSpacing()

    with tempfile.TemporaryDirectory(prefix="d015_probs_") as tmp:
        tmp = Path(tmp)
        ct_p = tmp / "case_0000.nii.gz"
        sitk.WriteImage(ct_lps, str(ct_p))

        sacrum_dir = Path(
            "/home/schoaq/taehee/nnUNet_data/results/Dataset006_PENGWIN_SacrumOnly"
            "/nnUNetTrainer__nnUNetPlans__3d_fullres"
        )
        pred_a = init_predictor(str(sacrum_dir))
        probs_a = predict_softmax(pred_a, ct_p, tmp / "s1")
        sacrum_mask = probs_a[1] > 0.5
        offsets = compute_d015_offset_channels(sacrum_mask, spacing)
        if offsets is None:
            offsets = np.zeros((3,) + sacrum_mask.shape, dtype=np.float32)
            side = np.zeros(sacrum_mask.shape, dtype=np.float32)
        else:
            side = (offsets[0] > 0).astype(np.float32)

        channels = [ct_p]
        for i, arr in enumerate(offsets, 1):
            p = tmp / f"case_{i:04d}.nii.gz"
            save_temp_channel(arr, ct_lps, p)
            channels.append(p)
        sm = tmp / "case_0004.nii.gz"
        save_temp_channel(side, ct_lps, sm)
        channels.append(sm)

        from scripts.inference_v7_d015 import predict_argmax_multichannel

        pred_b = init_predictor(str(results_dir))
        _ = predict_argmax_multichannel(pred_b, channels, tmp / "s2")
        npz_list = list((tmp / "s2").glob("*.npz"))
        if not npz_list:
            raise RuntimeError("no npz from D015 predict")
        with np.load(npz_list[0]) as data:
            probs = data["probabilities"].astype(np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"PENGWIN_{case_id}.npz"
    np.savez_compressed(out_path, probabilities=probs)
    print(f"[ok] {case_id} -> {out_path}  shape={probs.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=Path, default=DEFAULT_RAW / "imagesTr")
    ap.add_argument("--out_dir", type=Path, default=_ROOT / "predictions" / "d015_probs_train")
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--split", choices=("heldout", "all"), default="heldout")
    args = ap.parse_args()

    if args.cases:
        cases = args.cases
    elif args.split == "heldout":
        cases = list(HELDOUT)
    else:
        cases = [
            p.name.replace("PENGWIN_", "").replace("_0000.nii.gz", "")
            for p in sorted(args.images_dir.glob("PENGWIN_*_0000.nii.gz"))
        ]

    for cid in cases:
        try:
            export_case(cid, args.images_dir, args.out_dir, args.results_dir)
        except Exception as e:
            print(f"[fail] {cid}: {e}")


if __name__ == "__main__":
    main()
