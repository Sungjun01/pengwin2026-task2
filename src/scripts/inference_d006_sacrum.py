"""D006 best checkpoint 로 D003 imagesTr (170 pelvic case) 에 sliding-window inference 후
sacrum sigmoid 확률맵을 P_sacrum 채널 (Dataset007 의 channel 1) 로 저장.

실행 (GPU 5 지정):
    CUDA_VISIBLE_DEVICES=5 python scripts/inference_d006_sacrum.py

산출물:
    nnUNet_raw/Dataset007_PENGWIN_PelvicCascade/sacrum_probs/PENGWIN_XXX.nii.gz  (float32)

스크립트는 predict_from_files_sequential 의 임시 출력 (_seg.nii.gz + .npz) 을 가져와
.npz 의 softmax channel 1 (sacrum) 만 추출하고, 임시 파일은 정리한다.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


def main():
    nnunet_raw = Path(env("nnUNet_raw"))
    nnunet_results = Path(env("nnUNet_results"))

    model_folder = nnunet_results / "Dataset006_PENGWIN_SacrumOnly" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    input_folder = nnunet_raw / "Dataset003_PENGWIN_PelvicAnatomy" / "imagesTr"

    out_root = nnunet_raw / "Dataset007_PENGWIN_PelvicCascade"
    out_probs = out_root / "sacrum_probs"
    out_tmp = out_root / "_d006_predict_tmp"
    out_probs.mkdir(parents=True, exist_ok=True)
    out_tmp.mkdir(parents=True, exist_ok=True)

    print(f"[setup] model_folder    = {model_folder}")
    print(f"[setup] input_folder    = {input_folder}")
    print(f"[setup] output (probs)  = {out_probs}")
    print(f"[setup] output (tmp)    = {out_tmp}")
    print(f"[setup] CUDA visible    = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_folder),
        use_folds=("all",),
        checkpoint_name="checkpoint_best.pth",
    )

    t0 = time.time()
    predictor.predict_from_files_sequential(
        list_of_lists_or_source_folder=str(input_folder),
        output_folder_or_list_of_truncated_output_files=str(out_tmp),
        save_probabilities=True,
        overwrite=False,
        folder_with_segs_from_prev_stage=None,
    )
    print(f"[predict] sliding-window inference 완료, {time.time() - t0:.1f}s")

    case_ids = sorted(p.name.replace("_0000.nii.gz", "") for p in input_folder.glob("*_0000.nii.gz"))
    print(f"[postprocess] {len(case_ids)} cases 처리 시작")

    converted, skipped = 0, 0
    for case_id in case_ids:
        npz_path = out_tmp / f"{case_id}.npz"
        out_path = out_probs / f"{case_id}.nii.gz"

        if not npz_path.exists():
            print(f"  [skip] {case_id}: .npz 없음 ({npz_path})")
            skipped += 1
            continue
        if out_path.exists():
            converted += 1
            continue

        with np.load(npz_path) as data:
            keys = list(data.keys())
            if "probabilities" not in data:
                sys.exit(f"  [err] {case_id}: .npz 키 {keys} — 'probabilities' 없음")
            probs = data["probabilities"]

        if probs.ndim != 4 or probs.shape[0] < 2:
            sys.exit(f"  [err] {case_id}: probabilities shape {probs.shape} 예상과 다름")

        sacrum_prob = probs[1].astype(np.float32)

        src_ct = input_folder / f"{case_id}_0000.nii.gz"
        ct_img = sitk.ReadImage(str(src_ct))
        ct_arr_shape = sitk.GetArrayFromImage(ct_img).shape

        if sacrum_prob.shape != ct_arr_shape:
            sys.exit(
                f"  [err] {case_id}: sacrum_prob shape {sacrum_prob.shape} "
                f"vs CT shape {ct_arr_shape} 일치 안 함"
            )

        prob_img = sitk.GetImageFromArray(sacrum_prob)
        prob_img.CopyInformation(ct_img)
        sitk.WriteImage(prob_img, str(out_path))

        converted += 1
        if converted % 20 == 0:
            print(f"  [progress] {converted}/{len(case_ids)} 완료")

    print(f"[done] converted={converted} skipped={skipped} 총 {time.time() - t0:.1f}s")
    if converted > 0 and skipped == 0:
        print(f"[cleanup] 임시 폴더 {out_tmp} 삭제")
        shutil.rmtree(out_tmp, ignore_errors=True)
    else:
        print(f"[cleanup] converted={converted} skipped={skipped} — 안전 차원에서 임시 폴더 유지: {out_tmp}")


if __name__ == "__main__":
    main()
