"""Make D013's trainer class visible to nnU-Net's class finder.

`initialize_from_trained_model_folder` resolves the trainer by walking the FILESYSTEM under
nnunetv2/training/nnUNetTrainer, so injecting the class into a module namespace at runtime
does not work -- a .py file has to exist inside that package. The real trainer already ships
at /opt/algorithm/scripts/trainers/, so this is a one-line re-export placed where the finder
looks, not a second copy that could drift from it.
"""
import sys

if "/opt/algorithm" not in sys.path:
    sys.path.insert(0, "/opt/algorithm")

from scripts.trainers.nnUNetTrainerBorderWeightedFemurSnap import (  # noqa: F401
    nnUNetTrainerBorderWeightedFemurSnap,
)
