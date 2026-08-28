"""F-CIRL trainer with relaxed border weight (w=40) — diagnostic full-training run.

Identical to nnUNetTrainerBorderWeightedAseg (border-weighted DC+CE + Aseg loss,
spec §15) but BORDER_WEIGHT is lowered 139 -> 40. Motivation (recovery diagnosis
2026-05-26): the w139 D012 run showed held-out over-segmentation (sacrum over-seg
survived post-proc) and fold-level LH/RH core collapse (fold_4, fold_all). A milder
seam weight should reduce over-prediction / instability; F-CIRL adds the
boundary-uncertainty term on top. PELVIC ONLY (D012).
"""
from scripts.trainers.nnUNetTrainerBorderWeightedAseg import nnUNetTrainerBorderWeightedAseg


class nnUNetTrainerBorderWeightedAseg_w40(nnUNetTrainerBorderWeightedAseg):
    BORDER_WEIGHT = 40.0
