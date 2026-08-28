"""BorderTopo with halved topology weights (beta=0.05) — submission candidate arm.

Same losses as nnUNetTrainerBorderTopo but BETA_CLDICE/BETA_SEAMCORE lowered
0.1 -> 0.05. Motivation: at beta=0.1 the fold_0 run showed pseudo Dice ~0.06 below
the Aseg_w40 baseline at the same epoch, suggesting the topology terms may be over-
penalising semantic Dice. beta=0.05 is a gentler push toward border continuity.
Trained on fold_all for 1000 epochs as a production/submission candidate.
"""
from __future__ import annotations

from scripts.trainers.nnUNetTrainerBorderTopo import nnUNetTrainerBorderTopo


class nnUNetTrainerBorderTopo_b05(nnUNetTrainerBorderTopo):
    BETA_CLDICE = 0.05
    BETA_SEAMCORE = 0.05
