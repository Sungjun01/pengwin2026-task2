"""Border-weighted overfit trainer with milder weight (sqrt(139) ~= 12).

Escalation/tuning of nnUNetTrainerBorderWeighted50: weight 139 over-predicted the
border in easy/solitary regions (001 RH 1->7 over-seg). This milder weight tests
whether 12 reduces over-prediction while keeping the border learnable.
"""
from scripts.trainers.nnUNetTrainerBorderWeighted50 import nnUNetTrainerBorderWeighted50


class nnUNetTrainerBorderWeighted50_w12(nnUNetTrainerBorderWeighted50):
    BORDER_WEIGHT = 12.0
