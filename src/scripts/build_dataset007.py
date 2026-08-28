"""Dataset007_PENGWIN_PelvicCascade 빌드 — Phase B-1 cascade 의 Model_B (D007) 학습용.

입력:
    D003 imagesTr/                              -> channel 0 (CT)
    D007/sacrum_probs/PENGWIN_XXX.nii.gz        -> channel 1 (P_sacrum, D006 inference 결과)
    D003 labelsTr/                              -> 4-class anatomy label

산출:
    nnUNet_raw/Dataset007_PENGWIN_PelvicCascade/
        imagesTr/PENGWIN_XXX_0000.nii.gz   (CT, D003 에서 복사)
        imagesTr/PENGWIN_XXX_0001.nii.gz   (P_sacrum, sacrum_probs 에서 복사)
        labelsTr/PENGWIN_XXX.nii.gz        (D003 label 그대로)
        dataset.json
"""

import json
import os
import shutil
import sys
from pathlib import Path


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


def main():
    nnunet_raw = Path(env("nnUNet_raw"))

    d003 = nnunet_raw / "Dataset003_PENGWIN_PelvicAnatomy"
    d007 = nnunet_raw / "Dataset007_PENGWIN_PelvicCascade"

    src_img = d003 / "imagesTr"
    src_lbl = d003 / "labelsTr"
    src_prob = d007 / "sacrum_probs"

    dst_img = d007 / "imagesTr"
    dst_lbl = d007 / "labelsTr"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_prob.exists():
        sys.exit(f"sacrum_probs 폴더 없음: {src_prob}. inference_d006_sacrum.py 먼저 실행하세요.")

    case_ids = sorted(p.name.replace("_0000.nii.gz", "") for p in src_img.glob("*_0000.nii.gz"))
    print(f"[setup] {len(case_ids)} cases 처리")

    missing_probs = [c for c in case_ids if not (src_prob / f"{c}.nii.gz").exists()]
    missing_labels = [c for c in case_ids if not (src_lbl / f"{c}.nii.gz").exists()]
    if missing_probs:
        sys.exit(f"P_sacrum 누락 {len(missing_probs)} cases: {missing_probs[:5]}...")
    if missing_labels:
        sys.exit(f"label 누락 {len(missing_labels)} cases: {missing_labels[:5]}...")

    copied = 0
    for case_id in case_ids:
        # channel 0: CT
        src = src_img / f"{case_id}_0000.nii.gz"
        dst = dst_img / f"{case_id}_0000.nii.gz"
        if not dst.exists():
            shutil.copy2(src, dst)

        # channel 1: P_sacrum
        src = src_prob / f"{case_id}.nii.gz"
        dst = dst_img / f"{case_id}_0001.nii.gz"
        if not dst.exists():
            shutil.copy2(src, dst)

        # label
        src = src_lbl / f"{case_id}.nii.gz"
        dst = dst_lbl / f"{case_id}.nii.gz"
        if not dst.exists():
            shutil.copy2(src, dst)

        copied += 1

    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "P_sacrum",
        },
        "labels": {
            "background": 0,
            "sacrum": 1,
            "leftHip": 2,
            "rightHip": 3,
        },
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "Dataset007_PENGWIN_PelvicCascade",
        "description": (
            "PENGWIN Phase B-1 sacrum cascade Model_B 학습용. "
            "channel 0 = CT (D003 그대로), channel 1 = D006 best checkpoint inference 의 sacrum sigmoid 확률맵. "
            "label = D003 4-class anatomy. mirror augmentation 안전."
        ),
    }

    with open(d007 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[done] copied {copied} cases, dataset.json written at {d007 / 'dataset.json'}")


if __name__ == "__main__":
    main()
