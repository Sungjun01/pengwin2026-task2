"""PENGWIN Challenge_datasets 의 원본 1-200 instance label 을 170 pelvic case 모아 단일 디렉터리 구성.

용법.
    python scripts/build_gt_instance_dir.py --out_dir predictions/gt_instance
"""
import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--pelvic_range", nargs=2, type=int, default=[1, 200],
        help="pelvic case ID 범위 (1-200 default)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for part_dir in sorted(args.src_root.glob("PENGWIN26_task1_2_train_part*")):
        for case_dir in sorted(part_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            try:
                cid = int(case_dir.name)
            except ValueError:
                continue
            if not (args.pelvic_range[0] <= cid <= args.pelvic_range[1]):
                continue
            src = case_dir / "label.mha"
            if not src.exists():
                continue
            # nnUNet 의 prediction 은 PENGWIN_001.nii.gz 같은 이름. 일치시킴.
            out = args.out_dir / f"PENGWIN_{cid:03d}.nii.gz"
            if out.exists():
                continue
            img = sitk.ReadImage(str(src))
            arr = sitk.GetArrayFromImage(img).astype(np.uint16)
            new = sitk.GetImageFromArray(arr)
            new.CopyInformation(img)
            sitk.WriteImage(new, str(out))
            copied += 1
    print(f"[done] {copied} GT instance labels copied to {args.out_dir}")


if __name__ == "__main__":
    main()
