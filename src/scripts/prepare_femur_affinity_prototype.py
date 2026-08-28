"""Prepare commands for the D004 femur affinity fold_0 prototype.

This script does not start long training by default. It verifies the local D004
preprocessed dataset/trainer files and prints the exact commands for:
  1. installing the custom trainer into nnU-Net variants,
  2. launching fold_0 affinity training,
  3. comparing the checkpoint with the existing OOF femur 8-metric harness.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSED = Path("/home/schoaq/taehee/nnUNet_data/preprocessed")
DEFAULT_RESULTS = Path("/home/schoaq/taehee/nnUNet_data/results")
DATASET = "Dataset004_PENGWIN_FemurAnatomy"
TRAINER = "nnUNetTrainerFemurAffinity"
PLANS = "nnUNetPlans"
CONFIG = "3d_fullres"


def _nnunet_variants_dir() -> Path:
    import nnunetv2
    return Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer" / "variants"


def build_commands(
    preprocessed_dir: Path = DEFAULT_PREPROCESSED,
    results_dir: Path = DEFAULT_RESULTS,
    fold: int = 0,
    gpu: int = 6,
) -> list[str]:
    variants_dir = _nnunet_variants_dir()
    trainer_src = ROOT / "scripts" / "trainers" / f"{TRAINER}.py"
    affinity_ckpt_dir = results_dir / DATASET / f"{TRAINER}__{PLANS}__{CONFIG}"
    return [
        f"cp {trainer_src} {variants_dir}/",
        (
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu} "
            f"nnUNet_raw=/home/schoaq/taehee/nnUNet_data/raw "
            f"nnUNet_preprocessed={preprocessed_dir} "
            f"nnUNet_results={results_dir} "
            f"AFFINITY_LAMBDA=0.2 AFFINITY_POS_WEIGHT=1.0 AFFINITY_EPOCHS=500 "
            f"nnUNetv2_train {DATASET} {CONFIG} {fold} -tr {TRAINER} -p {PLANS}"
        ),
        (
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu} "
            f"PENGWIN_OOF_CKPT={affinity_ckpt_dir} PENGWIN_OOF_FOLD={fold} "
            "python3 -m scripts.oof_femur_8metric"
        ),
    ]


def check_environment(preprocessed_dir: Path, results_dir: Path) -> list[str]:
    problems: list[str] = []
    dataset_dir = preprocessed_dir / DATASET
    if not (dataset_dir / "dataset.json").exists():
        problems.append(f"missing {dataset_dir / 'dataset.json'}")
    if not (dataset_dir / "nnUNetPlans.json").exists():
        problems.append(f"missing {dataset_dir / 'nnUNetPlans.json'}")
    trainer_src = ROOT / "scripts" / "trainers" / f"{TRAINER}.py"
    if not trainer_src.exists():
        problems.append(f"missing {trainer_src}")
    if not results_dir.exists():
        problems.append(f"missing results dir {results_dir}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", type=Path, default=Path(os.environ.get("nnUNet_preprocessed", DEFAULT_PREPROCESSED)))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("nnUNet_results", DEFAULT_RESULTS)))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()

    problems = check_environment(args.preprocessed_dir, args.results_dir)
    if problems:
        print("[affinity-prototype] environment problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("[affinity-prototype] environment OK")
    print(f"[affinity-prototype] dataset={DATASET} trainer={TRAINER} fold={args.fold}")
    print("[affinity-prototype] commands:")
    for idx, command in enumerate(build_commands(args.preprocessed_dir, args.results_dir, args.fold, args.gpu), start=1):
        print(f"{idx}. {command}")


if __name__ == "__main__":
    main()
