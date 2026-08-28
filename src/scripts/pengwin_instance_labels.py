"""Utilities for PENGWIN bone-group instance label volumes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


BONE_GROUP_OFFSETS: dict[str, int] = {
    "sacrum": 0,
    "leftHip": 50,
    "rightHip": 100,
    "femur": 150,
}


def read_label_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to read {path}") from exc
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def write_label_volume(path: str | Path, label: np.ndarray, reference_path: str | Path | None = None) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, label.astype(np.uint16))
        return
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - SimpleITK is present in integration env
        raise ImportError(f"SimpleITK is required to write {path}") from exc
    image = sitk.GetImageFromArray(label.astype(np.uint16))
    if reference_path is not None:
        ref = sitk.ReadImage(str(reference_path))
        image.CopyInformation(ref)
    sitk.WriteImage(image, str(path))


def merge_bone_group_instance_labels(
    labels_by_bone: Mapping[str, np.ndarray],
    offsets: Mapping[str, int] = BONE_GROUP_OFFSETS,
) -> np.ndarray:
    """Merge local bone-group instance IDs into the global PENGWIN ID scheme."""
    if not labels_by_bone:
        raise ValueError("at least one bone-group label volume is required")
    shapes = {tuple(np.asarray(label).shape) for label in labels_by_bone.values()}
    if len(shapes) != 1:
        raise ValueError(f"all label volumes must have the same shape, got {sorted(shapes)}")

    shape = next(iter(shapes))
    merged = np.zeros(shape, dtype=np.uint16)
    for bone, label in labels_by_bone.items():
        if bone not in offsets:
            raise ValueError(f"unknown bone group {bone!r}; expected one of {sorted(offsets)}")
        arr = np.asarray(label)
        mask = arr > 0
        if not mask.any():
            continue
        if np.any(merged[mask] > 0):
            raise ValueError(f"overlapping nonzero voxels while merging bone group {bone!r}")
        merged[mask] = (arr[mask].astype(np.uint16) + int(offsets[bone])).astype(np.uint16)
    return merged


def merge_case_from_directory(
    labels_dir: str | Path,
    case_id: str,
    output_path: str | Path,
    file_ending: str = ".nii.gz",
    include_femur: bool = True,
) -> np.ndarray:
    labels_dir = Path(labels_dir)
    bones = ["sacrum", "leftHip", "rightHip"] + (["femur"] if include_femur else [])
    labels: dict[str, np.ndarray] = {}
    reference_path: Path | None = None
    for bone in bones:
        path = labels_dir / f"PENGWIN_{case_id}_{bone}{file_ending}"
        if not path.exists():
            continue
        labels[bone] = read_label_volume(path)
        reference_path = reference_path or path
    merged = merge_bone_group_instance_labels(labels)
    write_label_volume(output_path, merged, reference_path=reference_path if str(output_path).endswith((".nii.gz", ".mha")) else None)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--case-id", required=True, help="Case id without PENGWIN_ prefix, e.g. 004")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-ending", default=".nii.gz")
    parser.add_argument("--no-femur", action="store_true")
    args = parser.parse_args()
    merged = merge_case_from_directory(
        labels_dir=args.labels_dir,
        case_id=args.case_id,
        output_path=args.output,
        file_ending=args.file_ending,
        include_femur=not args.no_femur,
    )
    labels = [int(x) for x in np.unique(merged) if int(x) > 0]
    print({"shape": tuple(int(x) for x in merged.shape), "num_instances": len(labels), "labels": labels})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
