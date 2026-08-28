"""Full from-scratch Skeleton Recall — distinct output folder from the warm-start probe
(nnUNetTrainerSkeletonRecall). Same loss; only the class name differs so nnUNet writes its own
results dir. Used for the 1000-ep from-scratch run on fold_0 (held-out), with 50-ep snapshots +
checkpoint_best kept so we can gate the trajectory and pick the instance-best epoch (1000 epochs
is not always best — overfitting risk).
"""
from scripts.trainers.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall


class nnUNetTrainerSkeletonRecallFull(nnUNetTrainerSkeletonRecall):
    pass
