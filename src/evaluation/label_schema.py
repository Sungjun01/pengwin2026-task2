"""PENGWIN label scheme: 0 background, 1-50 sacrum, 51-100 leftHip,
101-150 rightHip, 151-200 femur (L/R merged per spec)."""
from __future__ import annotations

import numpy as np


BONE_RANGES: dict[int, tuple[int, int]] = {
    1: (1, 50),
    2: (51, 100),
    3: (101, 150),
    4: (151, 200),
}

ANATOMY_NAMES: dict[int, str] = {
    0: "background", 1: "sacrum", 2: "leftHip", 3: "rightHip", 4: "femur",
}


def anatomy_id_from_fragment_id(frag_id: int) -> int:
    """Return anatomy class (0-4) for a fragment ID (0-200)."""
    if frag_id == 0:
        return 0
    for anat, (lo, hi) in BONE_RANGES.items():
        if lo <= frag_id <= hi:
            return anat
    raise ValueError(f"Fragment ID {frag_id} not in valid PENGWIN range [0, 200]")


def remap_to_anatomy(lbl: np.ndarray) -> np.ndarray:
    """Convert a fragment-ID volume to a 5-class anatomy volume (0-4)."""
    out = np.zeros_like(lbl, dtype=np.uint8)
    for anat, (lo, hi) in BONE_RANGES.items():
        out[(lbl >= lo) & (lbl <= hi)] = anat
    return out
