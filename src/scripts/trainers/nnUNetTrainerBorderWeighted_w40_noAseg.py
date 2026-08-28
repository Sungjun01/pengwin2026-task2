"""F-CIRL ISOLATION ablation: border-weighted (w=40) WITHOUT the Aseg term.

Identical to nnUNetTrainerBorderWeightedAseg_w40 (the v18 pelvic trainer) except
ASEG_BETA = 0.0 -> the F-CIRL boundary-uncertainty (Aseg) loss contributes nothing
and its std-head receives no gradient. This is the long-pending Run-4a arm: train
this and the with-Aseg twin under identical data/seed/epochs, then compare INSTANCE
RECALL / fragment separation (not just border dice). Tests the report-99 hypothesis
that F-CIRL (rewarding seam-uncertainty) reinforces under-segmentation.
"""
from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import nnUNetTrainerBorderWeightedAseg_w40


class nnUNetTrainerBorderWeighted_w40_noAseg(nnUNetTrainerBorderWeightedAseg_w40):
    ASEG_BETA = 0.0  # F-CIRL OFF (isolation arm)
