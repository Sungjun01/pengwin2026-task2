"""Dataset009_PENGWIN_OracleSacrum 빌드 — Task D (F2 Oracle Upper Bound).

D003 의 CT + GT sacrum mask (label == 1) 를 2-channel input 으로, 라벨은 D003 의
4-class anatomy 그대로. cascade 의 sacrum conditioning mechanism 자체가 valid 한지
*P_sacrum quality 의 한계와 분리해서* 측정하기 위한 oracle upper bound.

판별 logic (progress_report_17 §3 매트릭스).
    가설 1 (Model_A quality 부족): Task D 가 모든 subset 에서 saturate → P_sacrum 만
        깨끗하면 cascade conditioning 충분. Model_A 강화 또는 inference TTA 만으로 해결.
    가설 2 (CNN/patch architectural 한계): GT sacrum 줘도 mode collapse 가 부분적
        지속 → cascade conditioning mechanism 자체가 한계, Spatial Moments / explicit
        coordinate 필요.

Task F (D008 Spatial Moments) 가 epoch 82 만에 saturate 한 결과를 보면 가설 2 쪽이
강하지만, Task D 의 결과가 *Task F 와 비슷한 epoch 에 saturate* 한다면 cascade
mechanism 자체는 valid (단지 P_sacrum quality 가 충분치 못함) 라는 추가 해석 가능.
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


def main():
    nnunet_raw = Path(env("nnUNet_raw"))

    d003 = nnunet_raw / "Dataset003_PENGWIN_PelvicAnatomy"
    d009 = nnunet_raw / "Dataset009_PENGWIN_OracleSacrum"

    src_img = d003 / "imagesTr"
    src_lbl = d003 / "labelsTr"

    dst_img = d009 / "imagesTr"
    dst_lbl = d009 / "labelsTr"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(p.name.replace("_0000.nii.gz", "") for p in src_img.glob("*_0000.nii.gz"))
    print(f"[setup] {len(case_ids)} cases — Dataset009 oracle 빌드 시작")

    for idx, case_id in enumerate(case_ids):
        src_ct = src_img / f"{case_id}_0000.nii.gz"
        src_label = src_lbl / f"{case_id}.nii.gz"
        if not src_ct.exists() or not src_label.exists():
            sys.exit(f"필수 파일 없음: {src_ct} or {src_label}")

        dst_ct = dst_img / f"{case_id}_0000.nii.gz"
        dst_sacrum = dst_img / f"{case_id}_0001.nii.gz"
        dst_label = dst_lbl / f"{case_id}.nii.gz"

        if not dst_ct.exists():
            shutil.copy2(src_ct, dst_ct)
        if not dst_label.exists():
            shutil.copy2(src_label, dst_label)

        if not dst_sacrum.exists():
            label_img = sitk.ReadImage(str(src_label))
            label_arr = sitk.GetArrayFromImage(label_img)
            sacrum_mask = (label_arr == 1).astype(np.float32)
            sac_img = sitk.GetImageFromArray(sacrum_mask)
            sac_img.CopyInformation(label_img)
            sitk.WriteImage(sac_img, str(dst_sacrum))

        if (idx + 1) % 20 == 0:
            print(f"  [progress] {idx + 1}/{len(case_ids)} cases 완료")

    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "GT_sacrum",
        },
        "labels": {
            "background": 0,
            "sacrum": 1,
            "leftHip": 2,
            "rightHip": 3,
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "Dataset009_PENGWIN_OracleSacrum",
        "description": (
            "Task D F2 Oracle Upper Bound. channel 0 = CT, channel 1 = GT sacrum binary "
            "(D003 label == 1). label = D003 4-class anatomy. cascade conditioning "
            "mechanism 의 valid 여부를 P_sacrum quality 한계와 분리 측정."
        ),
    }

    with open(d009 / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"[done] {len(case_ids)} cases — Dataset009 raw 빌드 완료 at {d009}")


if __name__ == "__main__":
    main()
