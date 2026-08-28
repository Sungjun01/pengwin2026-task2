"""Evaluate and visualize Voxel Mamba PENGWIN predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.io_utils import load_label_volume
from evaluation.pengwin_eval import evaluate_case
from scripts.infer_voxel_mamba_case import infer_case_from_paths
from scripts.voxel_mamba_io import load_lps_volume


def evaluate_prediction_paths(
    pairs: dict[str, tuple[Path, Path]],
    output_json: str | Path,
) -> dict[str, object]:
    per_case = {
        case_id: evaluate_case(pred_path, gt_path, min_cc_mm3=None)
        for case_id, (pred_path, gt_path) in sorted(pairs.items())
    }
    metrics = [k for k in next(iter(per_case.values())).keys()] if per_case else []
    aggregate = {}
    for key in metrics:
        vals = np.array([case[key] for case in per_case.values() if not np.isnan(case[key])], dtype=np.float64)
        if vals.size:
            aggregate[f"{key}_mean"] = float(vals.mean())
    result = {"per_case": per_case, "aggregate": aggregate}
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def make_case_visualization(
    case_id: str,
    image_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    output_png: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    foreground = (gt_zyx > 0) | (pred_zyx > 0)
    z = int(np.median(np.argwhere(foreground)[:, 0])) if foreground.any() else image_zyx.shape[0] // 2
    ct = image_zyx[z]
    lo, hi = np.percentile(ct, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [("CT", None), ("GT", gt_zyx[z] > 0), ("Pred", pred_zyx[z] > 0)]
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(ct, cmap="gray", vmin=lo, vmax=hi)
        if mask is not None:
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45)
        ax.set_title(f"{case_id} {title} z={z}")
        ax.axis("off")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def evaluate_cases_from_raw(
    checkpoint_path: str | Path,
    case_dirs: list[Path],
    output_dir: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    device: str = "auto",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    vis_dir = output_dir / "visualizations"
    pairs: dict[str, tuple[Path, Path]] = {}
    for case_dir in case_dirs:
        case_id = case_dir.name
        image_path = case_dir / "image.mha"
        gt_path = case_dir / "label.mha"
        if not image_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"case {case_id} must contain image.mha and label.mha")
        pred_path = pred_dir / f"{case_id}.mha"
        infer_case_from_paths(checkpoint_path, image_path, pred_path, patch_size=patch_size, device=device)
        pairs[case_id] = (pred_path, gt_path)
        image = load_lps_volume(image_path, dtype=np.float32).array
        gt, _ = load_label_volume(gt_path)
        pred, _ = load_label_volume(pred_path)
        make_case_visualization(case_id, image, gt, pred, vis_dir / f"{case_id}.png")
    return evaluate_prediction_paths(pairs, output_dir / "metrics.json")


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be like 96,96,96")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-dir", action="append", required=True, help="Case folder with image.mha and label.mha")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate_cases_from_raw(
        checkpoint_path=args.checkpoint,
        case_dirs=[Path(p) for p in args.case_dir],
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        device=args.device,
    )
    print(json.dumps({"event": "done", "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
