"""Atlas-가중 FULL training (from-scratch, 2000ep) — 동작 동일, 이름만 달라 출력 dir 분리."""
from scripts.trainers.nnUNetTrainerGaussianSkeletonRecall import nnUNetTrainerGaussianSkeletonRecall


class nnUNetTrainerGaussianSkeletonRecallFull(nnUNetTrainerGaussianSkeletonRecall):
    pass
