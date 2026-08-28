"""Run a trained Gaussian-Mamba parser on one prepared primitive sample."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gaussian_mamba_decode import BoundaryRefineConfig, DecisionConfig, decode_gaussian_mamba_case
from scripts.gaussian_mamba_model import GaussianMambaParser
from scripts.gaussian_mamba_training import _infer_config_from_sample, load_mamba_sample_npz
from scripts.gaussian_primitives import GaussianPrimitiveSet, load_primitive_set


def _read_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to read {path}") from exc
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool)


def _write_volume(path: str | Path, volume: np.ndarray, reference_path: str | Path | None = None) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, volume)
        return
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to write {path}") from exc
    image = sitk.GetImageFromArray(volume.astype(np.uint16))
    if reference_path is not None:
        ref = sitk.ReadImage(str(reference_path))
        image.CopyInformation(ref)
    sitk.WriteImage(image, str(path))


def _ordered_primitive_set(primitive_set: GaussianPrimitiveSet, order: np.ndarray) -> GaussianPrimitiveSet:
    def ordered(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else values[order]

    return GaussianPrimitiveSet(
        centers_zyx=primitive_set.centers_zyx[order],
        covariances=primitive_set.covariances[order],
        features=primitive_set.features[order],
        support_counts=primitive_set.support_counts[order],
        feature_names=primitive_set.feature_names,
        fragment_ids=ordered(primitive_set.fragment_ids),
        anatomy_labels=ordered(primitive_set.anatomy_labels),
        anatomy_zone_labels=ordered(primitive_set.anatomy_zone_labels),
        anatomy_side_labels=ordered(primitive_set.anatomy_side_labels),
        structural_role_labels=ordered(primitive_set.structural_role_labels),
        rank_scores=ordered(primitive_set.rank_scores),
        gt_match_ids=ordered(primitive_set.gt_match_ids),
        rank_components=ordered(primitive_set.rank_components),
        rank_component_names=primitive_set.rank_component_names,
        severity_labels=ordered(primitive_set.severity_labels),
        importance_weights=ordered(primitive_set.importance_weights),
    )


def infer_case_from_paths(
    checkpoint_path: str | Path,
    primitive_path: str | Path,
    sample_path: str | Path,
    foreground_path: str | Path,
    output_path: str | Path,
    hidden_dim: int = 128,
    backend: str = "gru",
    device: str | torch.device = "cpu",
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    batch = load_mamba_sample_npz(sample_path, device=device)
    sample_np = np.load(sample_path, allow_pickle=True)
    primitive_set = load_primitive_set(primitive_path)
    order = sample_np["order"].astype(np.int64) if "order" in sample_np else np.arange(len(primitive_set.centers_zyx))
    primitive_set = _ordered_primitive_set(primitive_set, order)

    model = GaussianMambaParser(
        input_dim=batch.features.shape[-1],
        hidden_dim=hidden_dim,
        anatomy_config=_infer_config_from_sample(batch),
        backend=backend,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    unexpected = list(load_result.unexpected_keys)
    missing = [key for key in load_result.missing_keys if not key.startswith("instance_")]
    if unexpected or missing:
        raise RuntimeError(f"checkpoint is incompatible; missing={missing}, unexpected={unexpected}")
    model.eval()

    with torch.no_grad():
        outputs = model(batch.features, batch.centers_zyx, batch.edge_index)

    foreground = _read_mask(foreground_path)
    decoded = decode_gaussian_mamba_case(
        primitive_set,
        foreground_logits=outputs.foreground_logits[0].detach().cpu().numpy(),
        edge_index=batch.edge_index.detach().cpu().numpy(),
        affinity_logits=outputs.affinity_logits[0].detach().cpu().numpy() if outputs.affinity_logits.numel() else np.zeros((0,), dtype=np.float32),
        foreground_mask=foreground,
        config=DecisionConfig(preserve_important_representative=True),
        boundary_refine=BoundaryRefineConfig(enabled=False),
    )
    _write_volume(output_path, decoded.instance_mask, reference_path=foreground_path if str(foreground_path).endswith(".mha") else None)
    summary = {
        "checkpoint": str(checkpoint_path),
        "primitive_path": str(primitive_path),
        "sample_path": str(sample_path),
        "output_path": str(output_path),
        "feature_dim": int(batch.features.shape[-1]),
        "num_primitives": int(len(decoded.keep_mask)),
        "kept_tokens": int(np.sum(decoded.keep_mask)),
        "num_pred_instances": int(len([x for x in np.unique(decoded.instance_mask) if int(x) > 0])),
        "affinity_edges": int(decoded.edge_index.shape[0]),
    }
    if summary_path is not None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--primitives", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--foreground", required=True, help="Foreground mask path, .npy or .mha")
    parser.add_argument("--output", required=True, help="Output instance mask, .npy or .mha")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--backend", default="gru", choices=("gru", "mamba"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = infer_case_from_paths(
        checkpoint_path=args.checkpoint,
        primitive_path=args.primitives,
        sample_path=args.sample,
        foreground_path=args.foreground,
        output_path=args.output,
        hidden_dim=args.hidden_dim,
        backend=args.backend,
        device=args.device,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
