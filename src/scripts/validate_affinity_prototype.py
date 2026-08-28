"""Validation gate for the femur affinity research prototype.

The affinity branch must not be packaged or submitted until a fold_0 checkpoint
exists and OOF/proxy commands have been run. This script makes that decision
explicit and reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RESULTS = Path("/home/schoaq/taehee/nnUNet_data/results")
DATASET = "Dataset004_PENGWIN_FemurAnatomy"
TRAINER = "nnUNetTrainerFemurAffinity"
PLANS_CONFIG = "nnUNetPlans__3d_fullres"


def affinity_model_dir(results_dir: Path) -> Path:
    return results_dir / DATASET / f"{TRAINER}__{PLANS_CONFIG}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()

    model_dir = affinity_model_dir(args.results_dir)
    checkpoint = model_dir / f"fold_{args.fold}" / "checkpoint_best.pth"
    print(f"[affinity-validation] model_dir={model_dir}")
    print(f"[affinity-validation] checkpoint={checkpoint}")

    if not checkpoint.exists():
        print("[affinity-validation] verdict=NOT_READY")
        print("[affinity-validation] reason=fold checkpoint missing; do not package or submit affinity branch")
        print("[affinity-validation] next=run scripts/prepare_femur_affinity_prototype.py and train fold_0")
        return

    print("[affinity-validation] verdict=CHECKPOINT_FOUND")
    print("[affinity-validation] required OOF command:")
    print(
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu} "
        f"PENGWIN_OOF_CKPT={model_dir} PENGWIN_OOF_FOLD={args.fold} "
        "python3 -m scripts.oof_femur_8metric"
    )
    print("[affinity-validation] required proxy/public smoke:")
    print("Run proxy_gt_eval.py on 002/004/005 and public005 smoke before any packaging.")


if __name__ == "__main__":
    main()
