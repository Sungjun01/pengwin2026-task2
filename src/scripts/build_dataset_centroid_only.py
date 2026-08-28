"""V1 Centroid-only ablation 빌드 — Dataset010 Predicted + Dataset011 Oracle.

paper §5 의 5-way ablation 의 핵심 — Spatial Moments 의 minimal subset 으로 PCA 채널의
contribution 분리 측정.

Input channels (4 channel total).
    0 : CT
    1 : norm_dx  — sacrum centroid 로부터의 x 방향 거리 (case 의 가장 긴 dim 으로 normalize, clip [-1, 1])
    2 : norm_dy  — y 방향 (anterior-posterior)
    3 : norm_dz  — z 방향 (cranial-caudal)

Modes.
    use_oracle=False : Dataset010_PENGWIN_CentroidOnly_Predicted
                        D007 의 channel 1 (P_sacrum, sigmoid 확률맵) 의 mass center 사용
    use_oracle=True  : Dataset011_PENGWIN_CentroidOnly_Oracle
                        D003 의 label==1 (GT sacrum binary) 의 mass center 사용

5 fix 반영.
    1. Dataset ID 010/011 분기 (D008 SpatialMoments 와 충돌 회피)
    2. D003 directory name 정확히 (PelvicAnatomy)
    3. Label key D003 와 일치 (leftHip / rightHip, underscore X)
    4. Broadcasting 으로 meshgrid 메모리 ~90% 절약
    5. P_sacrum tolerance -1e-3 (numerical precision)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


def compute_normalized_centroid_distance_maps(prob_array, spacing_xyz):
    """확률 가중 mass center + voxel-wise distance map (normalized to [-1, 1]).

    Axis convention.
        SimpleITK GetSpacing: (x, y, z)
        SimpleITK GetArrayFromImage: (z, y, x)
        prob_array shape: (Z, Y, X) → spacing 매핑 시 [2], [1], [0] 사용.

    Memory-efficient broadcasting 으로 meshgrid 피함. case 당 ~50M voxel * 3 channels
    float32 = ~600 MB (단일 meshgrid → 1.5 GB).
    """
    centroid_z, centroid_y, centroid_x = center_of_mass(prob_array)

    if np.isnan(centroid_z) or prob_array.sum() < 1e-3:
        return None, None, None

    z_dim, y_dim, x_dim = prob_array.shape

    z_coords = (np.arange(z_dim, dtype=np.float32) - np.float32(centroid_z)) * np.float32(spacing_xyz[2])
    y_coords = (np.arange(y_dim, dtype=np.float32) - np.float32(centroid_y)) * np.float32(spacing_xyz[1])
    x_coords = (np.arange(x_dim, dtype=np.float32) - np.float32(centroid_x)) * np.float32(spacing_xyz[0])

    max_length = max(
        z_dim * spacing_xyz[2],
        y_dim * spacing_xyz[1],
        x_dim * spacing_xyz[0],
    )
    half_max = np.float32(max_length / 2.0)

    # Broadcasting 시점에 (Z, Y, X) 로 확장
    norm_dz = np.broadcast_to(
        np.clip(z_coords / half_max, -1.0, 1.0)[:, None, None], (z_dim, y_dim, x_dim),
    ).astype(np.float32)
    norm_dy = np.broadcast_to(
        np.clip(y_coords / half_max, -1.0, 1.0)[None, :, None], (z_dim, y_dim, x_dim),
    ).astype(np.float32)
    norm_dx = np.broadcast_to(
        np.clip(x_coords / half_max, -1.0, 1.0)[None, None, :], (z_dim, y_dim, x_dim),
    ).astype(np.float32)

    return norm_dx, norm_dy, norm_dz


def build(use_oracle: bool):
    nnunet_raw = Path(env("nnUNet_raw"))

    if use_oracle:
        ds_id = 11
        suffix = "Oracle"
    else:
        ds_id = 10
        suffix = "Predicted"
    new_ds_name = f"Dataset{ds_id:03d}_PENGWIN_CentroidOnly_{suffix}"
    new_dir = nnunet_raw / new_ds_name

    # 원본 데이터
    d003_dir = nnunet_raw / "Dataset003_PENGWIN_PelvicAnatomy"
    d007_dir = nnunet_raw / "Dataset007_PENGWIN_PelvicCascade"

    src_img = d003_dir / "imagesTr"   # CT
    src_lbl = d003_dir / "labelsTr"   # 4-class anatomy label
    src_prob = d007_dir / "imagesTr"  # D007 의 channel 1 (P_sacrum)

    dst_img = new_dir / "imagesTr"
    dst_lbl = new_dir / "labelsTr"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(p.name.replace("_0000.nii.gz", "") for p in src_img.glob("*_0000.nii.gz"))
    print(f"[setup] {len(case_ids)} cases — {new_ds_name} 빌드 시작")

    valid_count = 0
    skipped: list[str] = []

    for idx, case_id in enumerate(case_ids):
        src_ct = src_img / f"{case_id}_0000.nii.gz"
        src_label_path = src_lbl / f"{case_id}.nii.gz"

        if use_oracle:
            label_img = sitk.ReadImage(str(src_label_path))
            label_arr = sitk.GetArrayFromImage(label_img)
            sacrum_array = (label_arr == 1).astype(np.float32)
            spacing_xyz = label_img.GetSpacing()
            ref_img = label_img
        else:
            src_psac = src_prob / f"{case_id}_0001.nii.gz"
            psac_img = sitk.ReadImage(str(src_psac))
            sacrum_array = sitk.GetArrayFromImage(psac_img).astype(np.float32)
            # tolerance -1e-3 (sigmoid numerical precision)
            assert sacrum_array.min() >= -1e-3 and sacrum_array.max() <= 1.001, (
                f"{case_id}: P_sacrum 범위 이상 (min {sacrum_array.min()}, max {sacrum_array.max()})"
            )
            sacrum_array = np.clip(sacrum_array, 0.0, 1.0)
            spacing_xyz = psac_img.GetSpacing()
            ref_img = psac_img

        norm_dx, norm_dy, norm_dz = compute_normalized_centroid_distance_maps(sacrum_array, spacing_xyz)
        if norm_dx is None:
            print(f"  [skip] {case_id}: empty sacrum mask")
            skipped.append(case_id)
            continue

        # CT, label 복사
        dst_ct = dst_img / f"{case_id}_0000.nii.gz"
        dst_label = dst_lbl / f"{case_id}.nii.gz"
        if not dst_ct.exists():
            shutil.copy2(src_ct, dst_ct)
        if not dst_label.exists():
            shutil.copy2(src_label_path, dst_label)

        # channel 1, 2, 3 저장
        for c, arr in zip((1, 2, 3), (norm_dx, norm_dy, norm_dz)):
            out = dst_img / f"{case_id}_{c:04d}.nii.gz"
            if out.exists():
                continue
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(ref_img)
            sitk.WriteImage(img, str(out))

        valid_count += 1
        if (idx + 1) % 20 == 0:
            print(f"  [progress] {idx + 1}/{len(case_ids)} (valid {valid_count}, skipped {len(skipped)})")

    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "norm_dx",
            "2": "norm_dy",
            "3": "norm_dz",
        },
        "labels": {
            "background": 0,
            "sacrum": 1,
            "leftHip": 2,
            "rightHip": 3,
        },
        "numTraining": valid_count,
        "file_ending": ".nii.gz",
        "name": new_ds_name,
        "description": (
            f"Centroid-only ablation ({'Oracle GT sacrum' if use_oracle else 'Predicted P_sacrum'} 의 mass center). "
            "channel 0 = CT, channel 1-3 = sacrum centroid 기준 voxel 의 normalized world coord. "
            "D008 의 minimal subset — PCA 채널 contribution 분리 측정용."
        ),
    }
    with open(new_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {new_ds_name}: valid {valid_count} / total {len(case_ids)}")
    if skipped:
        print(f"  skipped (empty mask): {len(skipped)} → {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["predicted", "oracle", "both"], default="both")
    args = p.parse_args()

    if args.mode in ("predicted", "both"):
        build(use_oracle=False)
    if args.mode in ("oracle", "both"):
        build(use_oracle=True)


if __name__ == "__main__":
    main()
