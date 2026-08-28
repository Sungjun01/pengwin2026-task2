"""4-class anatomy semantic prediction → PENGWIN fragment instance encoding.

Input  : per-case nii.gz with values in {0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur}
         (4-class output of D003/D005/D007/D008 등 nnUNet inference)
Output : per-case nii.gz with PENGWIN 의 fragment instance ID encoding
            1-50    sacrum fragments
            51-100  leftHip fragments
            101-150 rightHip fragments
            151-200 femur fragments

Processing.
  1. per anatomy class — 3D connected component (6-connectivity)
  2. per component — voxel count × voxel_volume (mm³) 계산
  3. 500 mm³ 미만 component 제거 (PENGWIN 평가 제외 기준)
  4. 남은 component 들을 size 내림차순으로 정렬해 anatomy 의 BONE_RANGES 안에서 순차 ID
  5. 한 anatomy 의 component 수가 50 을 초과하면 size 상위 50 만 유지 (PENGWIN 표준)

용법.
    python scripts/postprocess_to_fragments.py \\
        --pred_dir <path/to/4class predictions> \\
        --out_dir  <path/to/instance encoded> \\
        --min_volume_mm3 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

ANATOMY_TO_RANGE = {
    1: (1, 50),     # sacrum
    2: (51, 100),   # leftHip
    3: (101, 150),  # rightHip
    4: (151, 200),  # femur
}


def split_to_fragments(
    anatomy_arr: np.ndarray,
    voxel_volume_mm3: float,
    min_volume_mm3: float = 500.0,
) -> np.ndarray:
    """Anatomy class label (0-4) volume → fragment instance ID volume (0-200)."""
    if anatomy_arr.ndim != 3:
        raise ValueError(f"3D mask expected, got shape {anatomy_arr.shape}")
    out = np.zeros_like(anatomy_arr, dtype=np.uint16)

    structure = np.ones((3, 3, 3), dtype=bool)
    for anat, (lo, hi) in ANATOMY_TO_RANGE.items():
        mask = (anatomy_arr == anat)
        if not mask.any():
            continue
        labels, n_components = cc_label(mask, structure=structure)
        if n_components == 0:
            continue

        # 각 component 의 voxel count + mm³ 계산
        comp_sizes: list[tuple[int, int]] = []
        for cid in range(1, n_components + 1):
            voxel_count = int((labels == cid).sum())
            comp_sizes.append((cid, voxel_count))

        # 500 mm³ filter + size 내림차순 정렬
        kept = [
            (cid, vc) for cid, vc in comp_sizes
            if vc * voxel_volume_mm3 >= min_volume_mm3
        ]
        kept.sort(key=lambda x: x[1], reverse=True)
        max_slots = hi - lo + 1
        kept = kept[:max_slots]

        # PENGWIN range 안에서 ID 할당
        for slot, (cid, _) in enumerate(kept):
            out[labels == cid] = lo + slot

    return out


def process_case(pred_path: Path, out_path: Path, min_volume_mm3: float) -> dict:
    img = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()
    voxel_volume_mm3 = float(sx * sy * sz)

    if arr.max() > 4:
        sys.exit(
            f"{pred_path}: max label {arr.max()} > 4 — anatomy 4-class 가 아닌 듯, 입력 확인 필요"
        )

    frag_arr = split_to_fragments(arr, voxel_volume_mm3, min_volume_mm3=min_volume_mm3)

    out_img = sitk.GetImageFromArray(frag_arr.astype(np.uint16))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

    counts = {}
    for anat, (lo, hi) in ANATOMY_TO_RANGE.items():
        unique_ids = sorted({int(v) for v in np.unique(frag_arr) if lo <= int(v) <= hi})
        counts[anat] = len(unique_ids)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="4-class anatomy prediction → PENGWIN fragment instance encoding",
    )
    parser.add_argument("-p", "--pred_dir", required=True, type=Path)
    parser.add_argument("-o", "--out_dir", required=True, type=Path)
    parser.add_argument("--min_volume_mm3", type=float, default=500.0)
    parser.add_argument("--ext", default=".nii.gz")
    args = parser.parse_args()

    if not args.pred_dir.exists():
        sys.exit(f"pred_dir not found: {args.pred_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(args.pred_dir.glob(f"*{args.ext}"))
    if not pred_files:
        sys.exit(f"no {args.ext} under {args.pred_dir}")

    print(f"[setup] {len(pred_files)} cases, min_volume={args.min_volume_mm3} mm³")
    totals = {anat: 0 for anat in ANATOMY_TO_RANGE}
    for i, pred_path in enumerate(pred_files):
        out_path = args.out_dir / pred_path.name
        counts = process_case(pred_path, out_path, args.min_volume_mm3)
        for anat, c in counts.items():
            totals[anat] += c
        if (i + 1) % 20 == 0:
            print(f"  [progress] {i + 1}/{len(pred_files)}")

    print(f"[done] {len(pred_files)} cases — instance fragments written to {args.out_dir}")
    print(f"[stats] total fragments per anatomy:")
    for anat, total in totals.items():
        avg = total / len(pred_files)
        anat_name = {1: "sacrum", 2: "leftHip", 3: "rightHip", 4: "femur"}[anat]
        print(f"         anatomy {anat} ({anat_name:>8}): {total:5d} total, {avg:.2f} avg/case")


if __name__ == "__main__":
    main()
