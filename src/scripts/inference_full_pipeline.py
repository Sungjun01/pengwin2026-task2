"""PENGWIN 2026 Task 1 — fully-automatic inference pipeline (single CT → 1-200 instance mha).

Pipeline.
    1. CT mha 읽기
    2. D006 (Model_A sacrum binary) inference  → P_sacrum sigmoid 확률맵
    3. centroid + PCA decompose (Spatial Moments)
    4. 8-channel input (CT + P_sacrum + 3 rel + 3 PCA) 생성
    5. D008 (Model_B with Spatial Moments) inference  → 4-class anatomy
    6. D004 (femur specialist) inference  → 2-class femur binary
    7. Post-processing:
         D008 의 sacrum/leftHip/rightHip → instance ID 1-50/51-100/101-150
         D004 의 femur → instance ID 151-200
         connected component + 500 mm³ filter + size-sorted ID assignment
    8. PENGWIN encoding mha 저장 (input 의 spacing/origin/direction 그대로)

Output encoding (PENGWIN 2026 spec).
    0       : background
    1-50    : sacrum fragments
    51-100  : leftHip fragments
    101-150 : rightHip fragments
    151-200 : femur fragments  (L+R merged in label encoding, anatomy-level 평가 시 분리)

용법.
    python scripts/inference_full_pipeline.py --input <ct.mha> --output <pred.mha>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy.ndimage import (
    center_of_mass,
    label as cc_label,
    distance_transform_edt,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

try:
    from scripts.classify_case_type import classify_image
except ImportError:
    from classify_case_type import classify_image


ANATOMY_TO_RANGE = {
    1: (1, 50),     # sacrum  (from D008 class 1)
    2: (51, 100),   # leftHip (from D008 class 2)
    3: (101, 150),  # rightHip (from D008 class 3)
    4: (151, 200),  # femur   (from D004 class 1)
}


def env_or(default: str) -> str:
    return os.environ.get("nnUNet_results", default)


def init_predictor(
    model_folder: str,
    checkpoint: str = "checkpoint_best.pth",
    use_mirroring: bool = False,
    tile_step_size: float = 0.7,
    use_gaussian: bool = False,
) -> nnUNetPredictor:
    """T4 16GB / 10 min budget 호환 default.

    mirroring=False : 8x forward pass 줄임 (TTA 제거)
    tile_step_size=0.7 : sliding window overlap 줄여 sliding count 감소
    use_gaussian=False : sliding window 의 gaussian-smoothing 제거 (시간 약 10-20% 추가 절감)
    """
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_folder,
        use_folds=("all",),
        checkpoint_name=checkpoint,
    )
    return predictor


def predict_softmax(predictor: nnUNetPredictor, ct_path: Path, tmp_out: Path) -> np.ndarray:
    """nnUNetPredictor 로 single case inference 후 probabilities (C, Z, Y, X) 반환."""
    tmp_out.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files_sequential(
        list_of_lists_or_source_folder=[[str(ct_path)]],
        output_folder_or_list_of_truncated_output_files=str(tmp_out),
        save_probabilities=True,
        overwrite=True,
        folder_with_segs_from_prev_stage=None,
    )
    case_id = ct_path.stem.replace(".nii", "")
    npz_path = tmp_out / f"{case_id}.npz"
    if not npz_path.exists():
        npz_candidates = list(tmp_out.glob("*.npz"))
        if not npz_candidates:
            raise RuntimeError(f"no .npz from predict_from_files_sequential in {tmp_out}")
        npz_path = npz_candidates[0]
    with np.load(npz_path) as data:
        probs = data["probabilities"]
    return probs


def predict_softmax_multichannel(predictor: nnUNetPredictor, channel_paths: list[Path], tmp_out: Path) -> np.ndarray:
    """Multi-channel input 의 nnUNet inference (D008 의 8-channel 등)."""
    tmp_out.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files_sequential(
        list_of_lists_or_source_folder=[[str(p) for p in channel_paths]],
        output_folder_or_list_of_truncated_output_files=str(tmp_out),
        save_probabilities=True,
        overwrite=True,
        folder_with_segs_from_prev_stage=None,
    )
    npz_candidates = list(tmp_out.glob("*.npz"))
    if not npz_candidates:
        raise RuntimeError(f"no .npz from multichannel inference in {tmp_out}")
    with np.load(npz_candidates[0]) as data:
        probs = data["probabilities"]
    return probs


def compute_centroid_offset_channels(
    p_sacrum: np.ndarray,
    spacing_xyz: tuple[float, float, float],
):
    """V1 (D010) 용 3 channel — sacrum centroid 기준 normalized voxel offset 만.

    Returns:
        (3, Z, Y, X) float32 array — (norm_dx, norm_dy, norm_dz).
        case dim 의 max length 로 normalize, clip [-1, 1].
    """
    binary = p_sacrum > 0.5
    if binary.sum() < 10:
        return None
    coords = np.argwhere(binary)
    centroid = coords.mean(axis=0)

    Z, Y, X = p_sacrum.shape
    sx, sy, sz = spacing_xyz
    z_coords = (np.arange(Z, dtype=np.float32) - np.float32(centroid[0])) * np.float32(sz)
    y_coords = (np.arange(Y, dtype=np.float32) - np.float32(centroid[1])) * np.float32(sy)
    x_coords = (np.arange(X, dtype=np.float32) - np.float32(centroid[2])) * np.float32(sx)
    max_length = max(Z * sz, Y * sy, X * sx)
    half_max = np.float32(max_length / 2.0)

    norm_dz = np.broadcast_to(np.clip(z_coords / half_max, -1.0, 1.0)[:, None, None], (Z, Y, X)).astype(np.float32)
    norm_dy = np.broadcast_to(np.clip(y_coords / half_max, -1.0, 1.0)[None, :, None], (Z, Y, X)).astype(np.float32)
    norm_dx = np.broadcast_to(np.clip(x_coords / half_max, -1.0, 1.0)[None, None, :], (Z, Y, X)).astype(np.float32)
    return np.stack([norm_dx, norm_dy, norm_dz], axis=0)


def compute_spatial_moments_channels(
    p_sacrum: np.ndarray,
    spacing_xyz: tuple[float, float, float],
):
    """centroid + PCA decomposition → 6 channel (3 rel + 3 PCA, voxel-varying)."""
    binary = p_sacrum > 0.5
    if binary.sum() < 10:
        return None
    coords = np.argwhere(binary)
    centroid = coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    shape = p_sacrum.shape
    Z, Y, X = shape
    z_coords = np.arange(Z, dtype=np.float32) - np.float32(centroid[0])
    y_coords = np.arange(Y, dtype=np.float32) - np.float32(centroid[1])
    x_coords = np.arange(X, dtype=np.float32) - np.float32(centroid[2])

    dim = np.array(shape, dtype=np.float32)
    rel_z = (z_coords[:, None, None] / dim[0]).repeat(Y, axis=1).repeat(X, axis=2)
    rel_y = (y_coords[None, :, None] / dim[1]).repeat(Z, axis=0).repeat(X, axis=2)
    rel_x = (x_coords[None, None, :] / dim[2]).repeat(Z, axis=0).repeat(Y, axis=1)

    # PCA projection — voxel position 의 PCA frame projection
    zz, yy, xx = np.meshgrid(
        np.arange(Z, dtype=np.float32), np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32),
        indexing="ij",
    )
    pos = np.stack([zz, yy, xx], axis=0)
    delta = pos - np.asarray(centroid, dtype=np.float32).reshape(3, 1, 1, 1)
    pca = np.tensordot(eigenvectors.T.astype(np.float32), delta, axes=([1], [0]))
    scale = np.sqrt(np.asarray(eigenvalues, dtype=np.float32) + 1e-6).reshape(3, 1, 1, 1)
    pca = pca / scale

    return np.stack([rel_z, rel_y, rel_x, pca[0], pca[1], pca[2]], axis=0).astype(np.float32)


def save_temp_channel(arr: np.ndarray, ref_img: sitk.Image, path: Path) -> None:
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, str(path))


def per_anatomy_instance_assign(
    semantic_arr: np.ndarray,
    anatomy_class: int,
    target_range: tuple[int, int],
    voxel_volume_mm3: float,
    min_volume_mm3: float = 500.0,   # match eval omission threshold (was 1000 = forfeited ~10% recall ceiling)
    min_peak_distance: int = 20,
) -> np.ndarray:
    """semantic_arr 의 anatomy class voxel → watershed-based fragment split → PENGWIN range ID.

    V4→V5 변경 (preliminary 5-case submission 결과 fragment Dice 0.242 reactive fix).
    Single connected component + 500 mm³ filter 가 under-segmentation (한 anatomy 의
    fragment 들을 큰 덩어리로 묶어 instance recall 0.234) → distance transform + local
    maxima watershed 으로 강제 fragment 분리.

    Parameter calibration (PENGWIN_001 의 GT 6 fragment, dataset median 9):
      min_peak=20 voxel (≈16mm at 0.78mm spacing). min_vol lowered 1000→500 mm³ to
      match the eval omission threshold (1000 dropped ~10% of eval-scored fragments
      on the training distribution → recall loss).
    """
    mask = (semantic_arr == anatomy_class)
    if not mask.any():
        return np.zeros_like(semantic_arr, dtype=np.uint16)

    distance = distance_transform_edt(mask)
    peak_coords = peak_local_max(
        distance,
        min_distance=min_peak_distance,
        labels=mask.astype(np.int32),
        exclude_border=False,
    )

    if len(peak_coords) == 0:
        labels, _ = cc_label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    else:
        markers = np.zeros(mask.shape, dtype=np.int32)
        for idx, coord in enumerate(peak_coords, start=1):
            markers[tuple(coord)] = idx
        labels = watershed(-distance, markers=markers, mask=mask)

    n = int(labels.max())
    if n == 0:
        return np.zeros_like(semantic_arr, dtype=np.uint16)

    sizes = [(cid, int((labels == cid).sum())) for cid in range(1, n + 1)]
    kept = [(cid, sz) for cid, sz in sizes if sz * voxel_volume_mm3 >= min_volume_mm3]
    kept.sort(key=lambda x: x[1], reverse=True)

    lo, hi = target_range
    max_slots = hi - lo + 1
    kept = kept[:max_slots]

    out = np.zeros_like(semantic_arr, dtype=np.uint16)
    for slot, (cid, _) in enumerate(kept):
        out[labels == cid] = lo + slot
    return out


def run_pipeline(
    input_ct: Path,
    output_path: Path,
    nnunet_results: Path,
    tmp_root: Path | None = None,
    skip_femur: bool = False,
    stage2_variant: str = "v1",  # "v1" = D010 (4-ch) / "d008" = D008 (8-ch)
    tile_step_size: float = 1.0,
) -> dict:
    t_start = time.time()

    # 0. Tmp dir
    tmp_ctx = tempfile.TemporaryDirectory(prefix="pengwin_pipeline_") if tmp_root is None else None
    tmp = Path(tmp_ctx.name) if tmp_ctx else tmp_root
    tmp.mkdir(parents=True, exist_ok=True)

    # 1. Read CT + standardize to _0000.nii.gz
    ct_img = sitk.ReadImage(str(input_ct))
    spacing_xyz = ct_img.GetSpacing()
    voxel_volume_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    ct_arr = sitk.GetArrayFromImage(ct_img)
    case_id = "case"
    nnunet_ct_path = tmp / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(ct_img, str(nnunet_ct_path))

    # 1b. Case-type routing — decided upfront from image metadata (transverse FOV).
    #     Replaces the old post-hoc 50cm³ pelvic-volume heuristic, which ran the
    #     pelvic specialist on every case and mis-routed femur cases (→ dice 0).
    case_type = classify_image(ct_img)
    fov_x = ct_img.GetSpacing()[0] * ct_img.GetSize()[0]
    print(f"[router] case_type={case_type}  (transverse FOV {fov_x:.1f} mm)")
    is_pelvic_case = case_type == "pelvic"

    if is_pelvic_case:
        # 2. D006 Model_A inference → P_sacrum
        d006_dir = nnunet_results / "Dataset006_PENGWIN_SacrumOnly" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        predictor_a = init_predictor(str(d006_dir), tile_step_size=tile_step_size)
        out_a = tmp / "stage1_modela"
        probs_a = predict_softmax(predictor_a, nnunet_ct_path, out_a)
        p_sacrum = probs_a[1].astype(np.float32)  # channel 1 = sacrum
        del predictor_a
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 3. Stage 2 — V1 (D010, 4-ch) or D008 (8-ch) 분기
        if stage2_variant == "v1":
            offsets = compute_centroid_offset_channels(p_sacrum, spacing_xyz)
            if offsets is None:
                print("[warn] sacrum mask too small — fallback zeros")
                offsets = np.zeros((3,) + p_sacrum.shape, dtype=np.float32)
            v1_inputs = []
            for c in range(4):
                path = tmp / f"{case_id}_v1_{c:04d}.nii.gz"
                if c == 0:
                    shutil.copy2(nnunet_ct_path, path)
                else:
                    save_temp_channel(offsets[c - 1], ct_img, path)
                v1_inputs.append(path)
            d010_dir = nnunet_results / "Dataset010_PENGWIN_CentroidOnly_Predicted" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            predictor_b = init_predictor(str(d010_dir), tile_step_size=tile_step_size)
            out_b = tmp / "stage2_v1"
            probs_b = predict_softmax_multichannel(predictor_b, v1_inputs, out_b)
        else:  # "d008" — 8-channel full Spatial Moments
            moments = compute_spatial_moments_channels(p_sacrum, spacing_xyz)
            if moments is None:
                print("[warn] sacrum mask too small — fallback zeros")
                moments = np.zeros((6,) + p_sacrum.shape, dtype=np.float32)
            d008_inputs = []
            for c in range(8):
                path = tmp / f"{case_id}_d008_{c:04d}.nii.gz"
                if c == 0:
                    shutil.copy2(nnunet_ct_path, path)
                elif c == 1:
                    save_temp_channel(p_sacrum, ct_img, path)
                else:
                    save_temp_channel(moments[c - 2], ct_img, path)
                d008_inputs.append(path)
            d008_dir = nnunet_results / "Dataset008_PENGWIN_SpatialMoments" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            predictor_b = init_predictor(str(d008_dir), tile_step_size=tile_step_size)
            out_b = tmp / "stage2_modelb"
            probs_b = predict_softmax_multichannel(predictor_b, d008_inputs, out_b)

        pelvic_semantic = probs_b.argmax(axis=0).astype(np.uint8)
        del predictor_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # femur case — skip the pelvic pipeline (D006 → stage2) entirely to stay
        # within the T4 wall-time budget; only D004 (below) runs.
        pelvic_semantic = np.zeros(ct_arr.shape, dtype=np.uint8)

    femur_semantic = np.zeros_like(pelvic_semantic, dtype=np.uint8)
    if not skip_femur and not is_pelvic_case:
        d004_dir = nnunet_results / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        if d004_dir.exists():
            predictor_f = init_predictor(str(d004_dir))
            out_f = tmp / "stage3_femur"
            probs_f = predict_softmax(predictor_f, nnunet_ct_path, out_f)
            femur_semantic = (probs_f.argmax(axis=0) == 1).astype(np.uint8)
            del predictor_f
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"[warn] D004 femur model 없음 ({d004_dir}), femur 분절 skip")
    elif is_pelvic_case:
        print("[router] pelvic case, D004 skip")

    # 7. Post-processing — anatomy → PENGWIN instance encoding
    out_arr = np.zeros_like(pelvic_semantic, dtype=np.uint16)
    for anat_class, anat_range in [(1, ANATOMY_TO_RANGE[1]),
                                    (2, ANATOMY_TO_RANGE[2]),
                                    (3, ANATOMY_TO_RANGE[3])]:
        out_arr += per_anatomy_instance_assign(
            pelvic_semantic, anat_class, anat_range, voxel_volume_mm3,
        )
    out_arr += per_anatomy_instance_assign(
        femur_semantic, 1, ANATOMY_TO_RANGE[4], voxel_volume_mm3,
    )

    # 8. Save with input's metadata.
    #    grand-challenge.org 의 *Segmentation* kind 는 uint8 + MET_UCHAR 표준 —
    #    uint16 으로 쓰면 "Image segments could not be determined" reject.
    #    PENGWIN 1-200 ID 는 uint8 (0-255) 안 충분히 들어감.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(out_arr.astype(np.uint8))
    out_img.CopyInformation(ct_img)
    out_img = sitk.Cast(out_img, sitk.sitkUInt8)
    sitk.WriteImage(out_img, str(output_path), useCompression=True)

    elapsed = time.time() - t_start
    print(f"[done] {input_ct.name} → {output_path.name}  elapsed {elapsed:.1f}s")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()

    return {"elapsed_s": elapsed, "n_fragments": int(out_arr.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="default = $nnUNet_results")
    parser.add_argument("--tmp", type=Path, default=None,
                        help="default = system tempdir (auto cleanup)")
    parser.add_argument("--skip_femur", action="store_true",
                        help="D004 femur model inference 건너뛰기 (debugging 용)")
    parser.add_argument("--stage2", choices=["v1", "d008"], default="v1",
                        help="v1 = D010 (4-ch centroid offset, faster) / d008 = D008 (8-ch full Spatial Moments)")
    parser.add_argument("--tile_step_size", type=float, default=1.0,
                        help="nnUNet sliding window step size (1.0 = no overlap, 0.5 = default)")
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = Path(env_or("/home/schoaq/taehee/nnUNet_data/results"))

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    result = run_pipeline(
        input_ct=args.input,
        output_path=args.output,
        nnunet_results=args.results_dir,
        tmp_root=args.tmp,
        skip_femur=args.skip_femur,
        stage2_variant=args.stage2,
        tile_step_size=args.tile_step_size,
    )
    print(f"[stats] elapsed = {result['elapsed_s']:.1f}s, max fragment ID = {result['n_fragments']}")


if __name__ == "__main__":
    main()
