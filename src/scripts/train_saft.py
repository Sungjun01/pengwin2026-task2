"""Train SAFT-GD, a sacrum-anchored Transformer for PENGWIN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

from scripts.saft_coordinates import build_anchor_channels
from scripts.saft_losses import compute_saft_loss
from scripts.saft_model import SAFTSegmenter
from scripts.saft_signal_channels import build_signal_channels
from scripts.voxel_mamba_dataset import VoxelMambaCase, VoxelPatchSampler, discover_raw_cases, load_raw_case

RawEntry = tuple[str, Path, Path]


def build_saft_input_from_patch(
    image_czyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    signal_channels: bool = False,
) -> np.ndarray:
    """Concatenate the CT channel with 5 GT-free anchor priors derived from the CT.

    The anchor is built from the image only (`build_anchor_channels`), so the
    identical input is reproducible at inference where no label exists.
    """
    priors = build_anchor_channels(image_czyx[0], spacing_zyx)
    parts = [image_czyx.astype(np.float32, copy=False), priors]
    if signal_channels:
        parts.append(build_signal_channels(image_czyx[0], spacing_zyx))
    return np.concatenate(parts, axis=0)


def _normalize_split_case_id(case_name: str) -> str:
    return case_name.replace("PENGWIN_", "").strip()


def load_split_case_ids(
    split_json_paths: Sequence[str | Path],
    fold: int = 0,
    split: str = "train",
) -> set[str]:
    ids: set[str] = set()
    for path in split_json_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if fold >= len(data):
            raise IndexError(f"fold {fold} not found in {path}")
        if split not in data[fold]:
            raise KeyError(f"split {split!r} not found in fold {fold} of {path}")
        ids.update(_normalize_split_case_id(str(x)) for x in data[fold][split])
    return ids


def filter_entries_by_split(entries: Sequence[RawEntry], case_ids: set[str]) -> list[RawEntry]:
    return [entry for entry in entries if _normalize_split_case_id(entry[0]) in case_ids]


def _device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_saft(
    cases: Sequence[VoxelMambaCase],
    output_dir: str | Path,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    backbone: str = "swin_unetr",
    feature_channels: int = 24,
    device: str | torch.device = "auto",
    seed: int = 20260609,
    amp: bool = True,
    lr: float = 2e-4,
    signal_channels: bool = False,
) -> dict[str, object]:
    if not cases:
        raise ValueError("cases must not be empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = VoxelPatchSampler(cases, patch_size=patch_size, foreground_probability=0.75, seed=seed)
    in_channels = 9 if signal_channels else 6
    model = SAFTSegmenter(in_channels=in_channels, backbone=backbone, feature_channels=feature_channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device_obj.type == "cuda"))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    start = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        last_components: dict[str, float] = {}
        for _ in range(int(steps_per_epoch)):
            sample = sampler.sample_patch()
            x_np = build_saft_input_from_patch(sample.image, sample.spacing_zyx, signal_channels=signal_channels)
            x = torch.from_numpy(x_np)[None].to(device_obj)
            anatomy = torch.from_numpy(sample.anatomy_label.astype(np.int64, copy=False))[None].to(device_obj)
            fragments = torch.from_numpy(sample.fragment_label.astype(np.int64, copy=False))[None].to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device_obj.type == "cuda")):
                outputs = model(x)
                loss, components = compute_saft_loss(outputs, anatomy, fragments)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SAFT loss: {loss}")
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
                "backbone": backbone,
                "feature_channels": int(feature_channels),
                "patch_size": list(patch_size),
                "in_channels": int(in_channels),
                "signal_channels": bool(signal_channels),
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


class StreamingSAFTSampler:
    def __init__(
        self,
        entries: Sequence[RawEntry],
        patch_size: tuple[int, int, int],
        foreground_probability: float = 0.75,
        seed: int = 20260609,
        cache_size: int = 4,
    ) -> None:
        if not entries:
            raise ValueError("entries must not be empty")
        self.entries = list(entries)
        self.patch_size = patch_size
        self.foreground_probability = foreground_probability
        self.rng = random.Random(seed)
        self.cache_size = int(cache_size)
        self.cache: dict[str, VoxelMambaCase] = {}
        self.cache_order: list[str] = []

    def _load_case(self, entry: RawEntry) -> VoxelMambaCase:
        case_id, image_path, label_path = entry
        if case_id in self.cache:
            return self.cache[case_id]
        case = load_raw_case(case_id, image_path, label_path)
        self.cache[case_id] = case
        self.cache_order.append(case_id)
        while len(self.cache_order) > self.cache_size:
            old = self.cache_order.pop(0)
            self.cache.pop(old, None)
        return case

    def sample_patch(self):
        entry = self.entries[self.rng.randrange(len(self.entries))]
        case = self._load_case(entry)
        sampler = VoxelPatchSampler([case], patch_size=self.patch_size, foreground_probability=self.foreground_probability, seed=self.rng.randrange(2**31))
        return sampler.sample_patch()


def train_saft_streaming(
    entries: Sequence[RawEntry],
    output_dir: str | Path,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    backbone: str = "swin_unetr",
    feature_channels: int = 24,
    device: str | torch.device = "auto",
    seed: int = 20260609,
    amp: bool = True,
    cache_size: int = 4,
    lr: float = 2e-4,
    signal_channels: bool = False,
) -> dict[str, object]:
    if not entries:
        raise ValueError("entries must not be empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = StreamingSAFTSampler(entries, patch_size=patch_size, foreground_probability=0.75, seed=seed, cache_size=cache_size)
    in_channels = 9 if signal_channels else 6
    model = SAFTSegmenter(in_channels=in_channels, backbone=backbone, feature_channels=feature_channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and device_obj.type == "cuda"))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    start = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        last_components: dict[str, float] = {}
        for _ in range(int(steps_per_epoch)):
            sample = sampler.sample_patch()
            x_np = build_saft_input_from_patch(sample.image, sample.spacing_zyx, signal_channels=signal_channels)
            x = torch.from_numpy(x_np)[None].to(device_obj)
            anatomy = torch.from_numpy(sample.anatomy_label.astype(np.int64, copy=False))[None].to(device_obj)
            fragments = torch.from_numpy(sample.fragment_label.astype(np.int64, copy=False))[None].to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and device_obj.type == "cuda")):
                outputs = model(x)
                loss, components = compute_saft_loss(outputs, anatomy, fragments)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SAFT loss: {loss}")
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
                "backbone": backbone,
                "feature_channels": int(feature_channels),
                "patch_size": list(patch_size),
                "streaming": True,
                "num_entries": len(entries),
                "in_channels": int(in_channels),
                "signal_channels": bool(signal_channels),
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
    return {"epochs": int(epochs), "best_loss": best_loss, "output_dir": str(output_dir), "num_entries": len(entries)}


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be like 96,96,96")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(96, 96, 96))
    parser.add_argument("--backbone", choices=("swin_unetr", "tiny"), default="swin_unetr")
    parser.add_argument("--feature-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--streaming", action="store_true", help="Load cases on demand instead of preloading all volumes.")
    parser.add_argument("--split-json", action="append", default=[], help="nnUNet splits_final.json path; can be passed multiple times.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--signal-channels", action="store_true", help="Append CT-derived gradient/structure/Hessian channels.")
    args = parser.parse_args()
    entries = discover_raw_cases(args.raw_root)
    if not entries:
        raise FileNotFoundError(f"no raw cases found under {args.raw_root}")
    if args.split_json:
        ids = load_split_case_ids(args.split_json, fold=args.fold, split=args.split)
        entries = filter_entries_by_split(entries, ids)
        if not entries:
            raise RuntimeError(f"split selection produced no raw entries for {args.split_json}")
    if args.streaming:
        summary = train_saft_streaming(
            entries=entries,
            output_dir=args.output_dir,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            patch_size=args.patch_size,
            backbone=args.backbone,
            feature_channels=args.feature_channels,
            device=args.device,
            cache_size=args.cache_size,
            lr=args.lr,
            signal_channels=args.signal_channels,
        )
    else:
        cases = [load_raw_case(case_id, image_path, label_path) for case_id, image_path, label_path in entries]
        summary = train_saft(
            cases=cases,
            output_dir=args.output_dir,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            patch_size=args.patch_size,
            backbone=args.backbone,
            feature_channels=args.feature_channels,
            device=args.device,
            lr=args.lr,
            signal_channels=args.signal_channels,
        )
    print(json.dumps({"event": "done", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
