"""변형 비교용 trainer (동작 동일, 이름만 달라 출력 dir 분리 → 같은 fold 공정 비교)."""
from scripts.trainers.nnUNetTrainerGaussianSkeletonRecall import nnUNetTrainerGaussianSkeletonRecall


class nnUNetTrainerGaussianSkeletonRecallB(nnUNetTrainerGaussianSkeletonRecall):
    pass
