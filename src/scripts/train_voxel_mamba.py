"""Standalone training loop for full-voxel Mamba PENGWIN segmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

from scripts.voxel_mamba_dataset import (
    VoxelMambaCase,
    VoxelPatchSampler,
    discover_raw_cases,
    load_raw_case,
)
from scripts.voxel_mamba_losses import compute_voxel_mamba_loss
from scripts.voxel_mamba_model import VoxelMambaUNet


def _as_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sample_to_device(sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = sample["image"][None].to(device)
    anatomy = sample["anatomy_label"][None].to(device)
    fragments = sample["fragment_label"][None].to(device)
    return image, anatomy, fragments


def train_voxel_mamba(
    cases: Sequence[VoxelMambaCase],
    output_dir: str | Path,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    base_channels: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "auto",
    seed: int = 20260609,
    amp: bool = True,
) -> dict[str, object]:
    if not cases:
        raise ValueError("cases must not be empty")
    device_obj = _as_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = VoxelPatchSampler(cases, patch_size=patch_size, foreground_probability=0.7, seed=seed)
    model = VoxelMambaUNet(base_channels=base_channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device_obj.type == "cuda"))
    history: list[dict[str, float]] = []
    start = time.time()
    best_loss = float("inf")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        last_components: dict[str, float] = {}
        for _ in range(int(steps_per_epoch)):
            sample = sampler.sample_patch().to_torch()
            image, anatomy, fragments = _sample_to_device(sample, device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device_obj.type == "cuda")):
                outputs = model(image)
                loss, components = compute_voxel_mamba_loss(outputs, anatomy, fragments)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite voxel mamba loss: {loss}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            last_components = components | {"grad_norm": grad_norm}
        mean_loss = float(np.mean(losses))
        summary = {
            "epoch": float(epoch),
            "mean_loss": mean_loss,
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            **{f"last_{k}": float(v) for k, v in last_components.items()},
        }
        history.append(summary)
        state = {
            "model_state_dict": model.state_dict(),
            "config": {
                "base_channels": int(base_channels),
                "patch_size": list(patch_size),
                "epochs": int(epochs),
                "steps_per_epoch": int(steps_per_epoch),
            },
            "summary": summary,
            "history": history,
        }
        torch.save(state, output_dir / "latest.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(state, output_dir / "best.pt")
        (output_dir / "history.json").write_text(
            json.dumps({"history": history, "best_loss": best_loss, "elapsed_s": time.time() - start}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"event": "epoch_done", **summary, "best_loss": best_loss}, sort_keys=True), flush=True)
    return {"epochs": int(epochs), "best_loss": best_loss, "output_dir": str(output_dir)}


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be like 96,96,96")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Root containing <case>/image.mha and label.mha")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    entries = discover_raw_cases(args.raw_root)
    if not entries:
        raise FileNotFoundError(f"no raw cases found under {args.raw_root}")
    cases = [load_raw_case(case_id, image_path, label_path) for case_id, image_path, label_path in entries]
    summary = train_voxel_mamba(
        cases=cases,
        output_dir=args.output_dir,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({"event": "done", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
