"""Standalone 3D patch sampling for full-voxel Mamba training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch

from evaluation.label_schema import remap_to_anatomy
from scripts.voxel_mamba_io import load_lps_volume, normalize_hu


@dataclass(frozen=True)
class VoxelMambaCase:
    case_id: str
    image: np.ndarray
    label: np.ndarray
    spacing_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class VoxelPatchSample:
    case_id: str
    image: np.ndarray
    fragment_label: np.ndarray
    anatomy_label: np.ndarray
    origin_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]

    def to_torch(self) -> dict[str, torch.Tensor | str | tuple[int, int, int] | tuple[float, float, float]]:
        return {
            "case_id": self.case_id,
            "image": torch.from_numpy(self.image.astype(np.float32, copy=False)),
            "fragment_label": torch.from_numpy(self.fragment_label.astype(np.int64, copy=False)),
            "anatomy_label": torch.from_numpy(self.anatomy_label.astype(np.int64, copy=False)),
            "origin_zyx": self.origin_zyx,
            "spacing_zyx": self.spacing_zyx,
        }


def fragment_to_anatomy_target(fragment_label: np.ndarray) -> np.ndarray:
    return remap_to_anatomy(fragment_label.astype(np.int32, copy=False)).astype(np.int64, copy=False)


def discover_raw_cases(root: str | Path) -> list[tuple[str, Path, Path]]:
    """Return `(case_id, image_path, label_path)` entries under a PENGWIN raw root."""
    root = Path(root)
    cases: list[tuple[str, Path, Path]] = []
    for image_path in sorted(root.rglob("image.mha")):
        case_dir = image_path.parent
        label_path = case_dir / "label.mha"
        if label_path.exists():
            cases.append((case_dir.name, image_path, label_path))
    if cases:
        return cases
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_path = case_dir / "image.mha"
        label_path = case_dir / "label.mha"
        if image_path.exists() and label_path.exists():
            cases.append((case_dir.name, image_path, label_path))
    return cases


def load_raw_case(case_id: str, image_path: str | Path, label_path: str | Path) -> VoxelMambaCase:
    image = load_lps_volume(image_path, dtype=np.float32)
    label = load_lps_volume(label_path, dtype=np.int32)
    if image.array.shape != label.array.shape:
        raise ValueError(f"image/label shape mismatch for {case_id}: {image.array.shape} vs {label.array.shape}")
    return VoxelMambaCase(
        case_id=case_id,
        image=normalize_hu(image.array),
        label=label.array.astype(np.int32, copy=False),
        spacing_zyx=image.spacing_zyx,
    )


def _crop_with_padding(
    array: np.ndarray,
    start: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    pad_value: float | int = 0,
) -> np.ndarray:
    z0, y0, x0 = start
    dz, dy, dx = patch_size
    shape = array.shape[-3:]
    out = np.full(array.shape[:-3] + patch_size, pad_value, dtype=array.dtype)
    src_start = (max(z0, 0), max(y0, 0), max(x0, 0))
    src_end = (min(z0 + dz, shape[0]), min(y0 + dy, shape[1]), min(x0 + dx, shape[2]))
    dst_start = (src_start[0] - z0, src_start[1] - y0, src_start[2] - x0)
    dst_end = (
        dst_start[0] + max(src_end[0] - src_start[0], 0),
        dst_start[1] + max(src_end[1] - src_start[1], 0),
        dst_start[2] + max(src_end[2] - src_start[2], 0),
    )
    if src_end[0] <= src_start[0] or src_end[1] <= src_start[1] or src_end[2] <= src_start[2]:
        return out
    out[(..., slice(dst_start[0], dst_end[0]), slice(dst_start[1], dst_end[1]), slice(dst_start[2], dst_end[2]))] = array[
        (..., slice(src_start[0], src_end[0]), slice(src_start[1], src_end[1]), slice(src_start[2], src_end[2]))
    ]
    return out


class VoxelPatchSampler:
    def __init__(
        self,
        cases: Sequence[VoxelMambaCase],
        patch_size: tuple[int, int, int] = (96, 96, 96),
        foreground_probability: float = 0.7,
        seed: int | None = None,
    ) -> None:
        if not cases:
            raise ValueError("cases must not be empty")
        self.cases = list(cases)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.foreground_probability = float(foreground_probability)
        self.rng = random.Random(seed)

    def _choose_case(self) -> VoxelMambaCase:
        return self.cases[self.rng.randrange(len(self.cases))]

    def _choose_center(self, case: VoxelMambaCase) -> tuple[int, int, int]:
        foreground = np.argwhere(case.label > 0)
        if foreground.size and self.rng.random() < self.foreground_probability:
            idx = self.rng.randrange(len(foreground))
            return tuple(int(x) for x in foreground[idx])
        return tuple(self.rng.randrange(max(int(s), 1)) for s in case.label.shape)

    def sample_patch(self) -> VoxelPatchSample:
        case = self._choose_case()
        center = self._choose_center(case)
        start = tuple(int(c - p // 2) for c, p in zip(center, self.patch_size))
        image_patch = _crop_with_padding(case.image, start, self.patch_size, pad_value=0.0)[None]
        label_patch = _crop_with_padding(case.label, start, self.patch_size, pad_value=0)
        anatomy = fragment_to_anatomy_target(label_patch)
        return VoxelPatchSample(
            case_id=case.case_id,
            image=image_patch.astype(np.float32, copy=False),
            fragment_label=label_patch.astype(np.int32, copy=False),
            anatomy_label=anatomy,
            origin_zyx=start,
            spacing_zyx=case.spacing_zyx,
        )
