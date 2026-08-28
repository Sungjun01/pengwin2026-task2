"""Sliding-window inference for standalone Voxel Mamba PENGWIN segmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.label_schema import BONE_RANGES
from scripts.voxel_mamba_io import load_lps_volume, normalize_hu, save_label_volume_like
from scripts.voxel_mamba_model import VoxelMambaUNet


def _starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    step = max(int(round(patch * (1.0 - overlap))), 1)
    starts = list(range(0, max(length - patch, 0) + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_predict(
    model: torch.nn.Module,
    image_czyx: torch.Tensor,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Return accumulated anatomy logits `(C, Z, Y, X)` for a full volume."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    if image_czyx.dim() != 4:
        raise ValueError(f"image must be CxZxYxX, got {tuple(image_czyx.shape)}")
    _, z, y, x = image_czyx.shape
    dz, dy, dx = patch_size
    logits_sum: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    with torch.no_grad():
        for z0 in _starts(z, dz, overlap):
            for y0 in _starts(y, dy, overlap):
                for x0 in _starts(x, dx, overlap):
                    patch = image_czyx[:, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx][None].to(device)
                    output = model(patch)
                    logits = output.anatomy_logits[0].float().cpu()
                    if logits_sum is None:
                        logits_sum = torch.zeros((logits.shape[0], z, y, x), dtype=torch.float32)
                        weight = torch.zeros((1, z, y, x), dtype=torch.float32)
                    logits_sum[:, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] += logits
                    weight[:, z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] += 1.0
    assert logits_sum is not None and weight is not None
    return logits_sum / torch.clamp(weight, min=1.0)


def anatomy_to_fragment_labels(anatomy_zyx: np.ndarray, min_voxels: int = 1) -> np.ndarray:
    """Convert dense anatomy classes into PENGWIN-range component instance IDs."""
    from scipy.ndimage import generate_binary_structure, label as cc_label

    pred = np.zeros_like(anatomy_zyx, dtype=np.int32)
    structure = generate_binary_structure(3, 3)
    for anatomy_id, (lo, hi) in BONE_RANGES.items():
        components, count = cc_label(anatomy_zyx == anatomy_id, structure=structure)
        next_id = int(lo)
        for comp_id in range(1, int(count) + 1):
            if next_id > hi:
                break
            mask = components == comp_id
            if int(mask.sum()) < int(min_voxels):
                continue
            pred[mask] = next_id
            next_id += 1
    return pred


def infer_case_from_paths(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    base_channels: int | None = None,
    device: str = "auto",
) -> Path:
    device_obj = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = VoxelMambaUNet(base_channels=int(base_channels or config.get("base_channels", 16)))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    volume = load_lps_volume(image_path, dtype=np.float32)
    image = torch.from_numpy(normalize_hu(volume.array)[None])
    logits = sliding_window_predict(model, image, patch_size=patch_size, device=device_obj)
    anatomy = torch.argmax(logits, dim=0).numpy().astype(np.int32)
    pred = anatomy_to_fragment_labels(anatomy)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_label_volume_like(pred, image_path, output_path)
    return output_path


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be like 96,96,96")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    out = infer_case_from_paths(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        device=args.device,
    )
    print(json.dumps({"event": "done", "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
