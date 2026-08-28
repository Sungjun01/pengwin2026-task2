"""Dataset008_PENGWIN_SpatialMoments 빌드 — Phase B-1 cascade Model_B + Spatial Moments.

D007 의 2-channel (CT + P_sacrum) 위에 sacrum 의 mass moment 기반 6 channel 을 추가.
첫 번째 motivation 은 D007 의 LH/RH mode collapse — patch-based CNN 이 absolute
spatial coordinate 를 학습 못 한다는 한계 (progress_report_19) — 를 입력 단에서
explicit 좌표 채널로 우회하는 것.

Input channel (총 8 channel).
    0: CT
    1: P_sacrum (D006 best checkpoint inference 의 sigmoid 확률맵)
    2-4: relative coord — (voxel_pos - sacrum_centroid) 를 case dim 으로 normalize
         (CoordConv 식 anatomy-tied 좌표, voxel 위치별 다른 값)
    5-7: PCA coord — sacrum 의 principal axis frame 에서의 voxel 좌표
         (sacrum 자체의 anatomical coordinate system 에서 voxel 이 어디인지)

label.
    D003 4-class (bg / sacrum / leftHip / rightHip) 그대로.

계산.
    sacrum_binary = P_sacrum > 0.5
    coords = argwhere(sacrum_binary)
    centroid = coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    # 각 voxel pos p 에 대해:
    rel_p = (p - centroid) / case_dim          # channels 2-4, 범위 ~ -0.5..0.5
    pca_p = eigenvectors.T @ (p - centroid) / sqrt(eigenvalues + eps)
                                                # channels 5-7, z-score 식 범위

normalization scheme (dataset.json 의 'normalization_schemes').
    CT          → CTNormalization
    P_sacrum    → ZScoreNormalization
    relative x  → NoNormalization (이미 -0.5..0.5)
    relative y  → NoNormalization
    relative z  → NoNormalization
    PCA_1       → NoNormalization (이미 z-score 식)
    PCA_2       → NoNormalization
    PCA_3       → NoNormalization
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


def compute_centroid_and_pca(prob_array: np.ndarray, threshold: float = 0.5):
    """sacrum 확률맵에서 mass center + covariance eigendecomposition.

    Returns:
        centroid: (3,) array in (z, y, x) order (numpy axis order matches SimpleITK array)
        eigenvalues: (3,) array
        eigenvectors: (3, 3) matrix, columns are eigenvectors
    """
    binary = prob_array > threshold
    coords = np.argwhere(binary)
    if coords.shape[0] < 10:
        sys.exit(f"sacrum voxel count {coords.shape[0]} 너무 적음, 빌드 중단")
    centroid = coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return centroid, eigenvalues, eigenvectors


def make_spatial_moment_channels(shape, centroid, eigenvalues, eigenvectors, eps: float = 1e-6):
    """Voxel-wise relative coord (3) + PCA coord (3) array 생성.

    Returns:
        (6, Z, Y, X) float32 array
    """
    Z, Y, X = shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    pos = np.stack([zz, yy, xx], axis=0).astype(np.float32)  # (3, Z, Y, X)

    # Relative coord normalized by case dimension
    dim = np.array(shape, dtype=np.float32).reshape(3, 1, 1, 1)
    rel = (pos - np.asarray(centroid, dtype=np.float32).reshape(3, 1, 1, 1)) / dim

    # PCA coord = eigenvectors.T @ (pos - centroid), then /sqrt(eigenvalue)
    delta = pos - np.asarray(centroid, dtype=np.float32).reshape(3, 1, 1, 1)  # (3, Z, Y, X)
    pca = np.tensordot(eigenvectors.T.astype(np.float32), delta, axes=([1], [0]))  # (3, Z, Y, X)
    scale = np.sqrt(np.asarray(eigenvalues, dtype=np.float32) + eps).reshape(3, 1, 1, 1)
    pca = pca / scale

    return np.concatenate([rel, pca], axis=0).astype(np.float32)


def save_channel_nii(channel_arr: np.ndarray, reference_img: sitk.Image, out_path: Path):
    img = sitk.GetImageFromArray(channel_arr)
    img.CopyInformation(reference_img)
    sitk.WriteImage(img, str(out_path))


def main():
    nnunet_raw = Path(env("nnUNet_raw"))

    d003 = nnunet_raw / "Dataset003_PENGWIN_PelvicAnatomy"
    d007 = nnunet_raw / "Dataset007_PENGWIN_PelvicCascade"
    d008 = nnunet_raw / "Dataset008_PENGWIN_SpatialMoments"

    src_img = d003 / "imagesTr"
    src_lbl = d003 / "labelsTr"
    src_prob = d007 / "sacrum_probs"

    dst_img = d008 / "imagesTr"
    dst_lbl = d008 / "labelsTr"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_prob.exists():
        sys.exit(f"sacrum_probs 폴더 없음: {src_prob}. inference_d006_sacrum.py 먼저 실행하세요.")

    case_ids = sorted(p.name.replace("_0000.nii.gz", "") for p in src_img.glob("*_0000.nii.gz"))
    print(f"[setup] {len(case_ids)} cases 처리 — Dataset008 빌드 시작")

    for idx, case_id in enumerate(case_ids):
        # 1. CT 와 label, P_sacrum 의 source 경로
        src_ct = src_img / f"{case_id}_0000.nii.gz"
        src_prob_path = src_prob / f"{case_id}.nii.gz"
        src_label_path = src_lbl / f"{case_id}.nii.gz"

        for required in (src_ct, src_prob_path, src_label_path):
            if not required.exists():
                sys.exit(f"필수 파일 없음: {required}")

        # 2. P_sacrum 로드, centroid + PCA 계산
        prob_img = sitk.ReadImage(str(src_prob_path))
        prob_arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
        shape = prob_arr.shape  # (Z, Y, X)

        centroid, eigenvalues, eigenvectors = compute_centroid_and_pca(prob_arr)

        # 3. Spatial moment channels 생성
        moments = make_spatial_moment_channels(shape, centroid, eigenvalues, eigenvectors)

        # 4. 저장 — channel 0,1 은 D007 그대로 복사, channel 2-7 은 새로 생성
        dst_paths = [dst_img / f"{case_id}_{i:04d}.nii.gz" for i in range(8)]

        # channel 0 (CT) — symlink to D003 imagesTr 의 _0000
        if not dst_paths[0].exists():
            shutil.copy2(src_ct, dst_paths[0])
        # channel 1 (P_sacrum) — symlink to D007 sacrum_probs
        if not dst_paths[1].exists():
            shutil.copy2(src_prob_path, dst_paths[1])

        # channels 2-7 — moments
        for c in range(6):
            if not dst_paths[c + 2].exists():
                save_channel_nii(moments[c], prob_img, dst_paths[c + 2])

        # label
        dst_label = dst_lbl / f"{case_id}.nii.gz"
        if not dst_label.exists():
            shutil.copy2(src_label_path, dst_label)

        if (idx + 1) % 20 == 0:
            print(f"  [progress] {idx + 1}/{len(case_ids)} cases 완료")

    # 5. dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "P_sacrum",
            "2": "rel_z",
            "3": "rel_y",
            "4": "rel_x",
            "5": "pca_1",
            "6": "pca_2",
            "7": "pca_3",
        },
        "labels": {
            "background": 0,
            "sacrum": 1,
            "leftHip": 2,
            "rightHip": 3,
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "Dataset008_PENGWIN_SpatialMoments",
        "description": (
            "Phase B-1 cascade + Spatial Moments (mechanism C1 변종). "
            "channel 0 = CT, channel 1 = P_sacrum, "
            "channel 2-4 = sacrum centroid 기준 relative coord (case dim normalized), "
            "channel 5-7 = sacrum 의 PCA coordinate frame projection. "
            "label = D003 4-class anatomy. progress_report_19 의 1차 카드 평가용."
        ),
    }

    with open(d008 / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"[done] {len(case_ids)} cases — Dataset008 raw 빌드 완료 at {d008}")


if __name__ == "__main__":
    main()
