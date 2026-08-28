"""Produce decoded_instances metrics by running model forward and decode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.pengwin_eval import evaluate_f_level
from scripts.infer_gaussian_mamba_case import infer_case_from_paths
from scripts.pengwin_instance_labels import merge_case_from_directory


def _read_label(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path), (1.0, 1.0, 1.0)
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to read {path}") from exc
    image = sitk.ReadImage(str(path))
    spacing_xyz = image.GetSpacing()
    return sitk.GetArrayFromImage(image), (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))


def _mean_metric(cases: list[dict[str, Any]], key: str) -> float:
    values = [float(c[key]) for c in cases if key in c and np.isfinite(float(c[key]))]
    return float(np.mean(values)) if values else float("nan")


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(cases),
        "RQ_F": _mean_metric(cases, "RQ_F"),
        "LDSC_F": _mean_metric(cases, "LDSC_F"),
        "HD95_F_mm": _mean_metric(cases, "HD95_F_mm"),
        "ASSD_F_mm": _mean_metric(cases, "ASSD_F_mm"),
        "mean_num_pred_instances": _mean_metric(cases, "num_pred_instances"),
        "mean_kept_tokens": _mean_metric(cases, "kept_tokens"),
    }


def produce_decoded_report(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    prediction_dir: str | Path,
    hidden_dim: int = 128,
    backend: str = "gru",
    device: str = "cpu",
    hard_case_ids: Iterable[str] = ("004", "064", "075", "084", "087"),
) -> dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    case_metrics: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        pred_path = prediction_dir / f"{case_id}_decoded.npy"
        gt_path = case.get("gt")
        if gt_path is None:
            if "gt_labels_dir" not in case:
                raise KeyError(f"case {case_id} must provide either 'gt' or 'gt_labels_dir'")
            gt_path = prediction_dir / f"{case_id}_merged_gt.npy"
            merge_case_from_directory(
                labels_dir=case["gt_labels_dir"],
                case_id=case_id,
                output_path=gt_path,
                file_ending=case.get("file_ending", ".nii.gz"),
                include_femur=bool(case.get("include_femur", True)),
            )
        infer_summary = infer_case_from_paths(
            checkpoint_path=checkpoint_path,
            primitive_path=case["primitives"],
            sample_path=case["sample"],
            foreground_path=case["foreground"],
            output_path=pred_path,
            hidden_dim=hidden_dim,
            backend=backend,
            device=device,
        )
        gt, spacing = _read_label(gt_path)
        pred, _ = _read_label(pred_path)
        metrics = evaluate_f_level(gt, pred, spacing=spacing)
        metrics.update(
            {
                "case_id": case_id,
                "gt_path": str(gt_path),
                "prediction_path": str(pred_path),
                "num_pred_instances": infer_summary["num_pred_instances"],
                "kept_tokens": infer_summary["kept_tokens"],
            }
        )
        case_metrics.append(metrics)
    hard_case_ids = {str(x) for x in hard_case_ids}
    hard_cases = {
        case_id: _summarize([m for m in case_metrics if m["case_id"] == case_id])
        for case_id in sorted(hard_case_ids)
        if any(m["case_id"] == case_id for m in case_metrics)
    }
    decoded_instances = _summarize(case_metrics)
    decoded_instances["cases"] = case_metrics
    decoded_instances["hard_cases"] = hard_cases
    report = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "decoded_instances": decoded_instances,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--backend", default="gru", choices=("gru", "mamba"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hard-cases", nargs="*", default=["004", "064", "075", "084", "087"])
    args = parser.parse_args()
    report = produce_decoded_report(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        prediction_dir=args.prediction_dir,
        hidden_dim=args.hidden_dim,
        backend=args.backend,
        device=args.device,
        hard_case_ids=args.hard_cases,
    )
    print(json.dumps(report["decoded_instances"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
