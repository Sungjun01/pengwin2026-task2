"""FROZEN copy of the task2-v5 click postprocessing (axis-flipped parse_clicks).

Kept ONLY for the OOF gate: lets us score the old behavior against the v6
self-correcting parser on the SAME v19 instance maps, to confirm the coordinate
fix is a net win / not a regression. Do not import this in the submission path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


ANATOMY_RANGES: dict[str, tuple[int, int]] = {
    "sacrum": (1, 50),
    "leftHip": (51, 100),
    "rightHip": (101, 150),
    "femur": (151, 200),
}


@dataclass(frozen=True)
class Click:
    anatomy: str
    point_zyx: tuple[int, int, int]
    name: str


def _anatomy_from_name(name: str) -> str:
    text = name.lower()
    if "sacrum" in text:
        return "sacrum"
    if "left" in text and ("hip" in text or "hipbone" in text):
        return "leftHip"
    if "right" in text and ("hip" in text or "hipbone" in text):
        return "rightHip"
    if "femur" in text:
        return "femur"
    raise ValueError(f"cannot infer anatomy from click name: {name}")


def _points_from_click_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        points: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                points.extend(item.get("points", []))
        return points
    if isinstance(data, dict):
        return list(data.get("points", []))
    return []


def parse_clicks(path_or_data: str | Path | dict[str, Any] | list[Any]) -> list[Click]:
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data) as f:
            data = json.load(f)
    else:
        data = path_or_data
    clicks: list[Click] = []
    for point in _points_from_click_payload(data):
        name = str(point["name"])
        x, y, z = (int(v) for v in point["point"])
        clicks.append(Click(_anatomy_from_name(name), (z, y, x), name))
    return clicks


def _clip_point(point_zyx: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(np.clip(p, 0, shape[i] - 1)) for i, p in enumerate(point_zyx))  # type: ignore[return-value]


def _nearest_label(mask: np.ndarray, point: tuple[int, int, int], labels: np.ndarray) -> int | None:
    if not mask.any():
        return None
    _, inds = distance_transform_edt(~mask, return_indices=True)
    nearest = tuple(int(inds[axis][point]) for axis in range(3))
    label = int(labels[nearest])
    return label if label != 0 else None


def _marker_for_click(mask: np.ndarray, point: tuple[int, int, int]) -> tuple[int, int, int]:
    if mask[point]:
        return point
    _, inds = distance_transform_edt(~mask, return_indices=True)
    return tuple(int(inds[axis][point]) for axis in range(3))  # type: ignore[return-value]


def _split_mask_by_clicks(mask: np.ndarray, points: list[tuple[int, int, int]]) -> np.ndarray:
    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, point in enumerate(points, start=1):
        markers[_marker_for_click(mask, point)] = idx
    if int(markers.max()) <= 1:
        return np.where(mask, markers.max(), 0).astype(np.int32)
    distance = distance_transform_edt(mask)
    return watershed(-distance, markers=markers, mask=mask).astype(np.int32)


def relabel_from_clicks(instance_arr: np.ndarray, clicks: list[Click]) -> np.ndarray:
    out = np.zeros_like(instance_arr, dtype=np.uint16)
    next_id = {anat: lo for anat, (lo, _) in ANATOMY_RANGES.items()}

    for anatomy in ANATOMY_RANGES:
        lo, hi = ANATOMY_RANGES[anatomy]
        anat_clicks = [c for c in clicks if c.anatomy == anatomy]
        if not anat_clicks:
            continue
        anat_mask = (instance_arr >= lo) & (instance_arr <= hi)
        selected: dict[int, list[tuple[int, int, int]]] = {}
        for click in anat_clicks:
            point = _clip_point(click.point_zyx, instance_arr.shape)
            label = int(instance_arr[point]) if anat_mask[point] else None
            if label is None or not (lo <= label <= hi):
                label = _nearest_label(anat_mask, point, instance_arr)
            if label is None or not (lo <= label <= hi):
                continue
            selected.setdefault(label, []).append(point)

        for source_label, points in selected.items():
            source_mask = instance_arr == source_label
            if not source_mask.any():
                continue
            if len(points) == 1:
                if next_id[anatomy] <= hi:
                    out[source_mask] = next_id[anatomy]
                    next_id[anatomy] += 1
                continue
            split = _split_mask_by_clicks(source_mask, points)
            for split_id in range(1, int(split.max()) + 1):
                if next_id[anatomy] > hi:
                    break
                part = split == split_id
                if part.any():
                    out[part] = next_id[anatomy]
                    next_id[anatomy] += 1
    return out
