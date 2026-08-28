"""Fast warm-start PROBE of the Supervoxel loss — separate results folder from the scratch run.

Identical loss to nnUNetTrainerSupervoxelTopo; only the class name differs so nnUNet writes to
its own output dir (.../nnUNetTrainerSupervoxelTopoProbe__.../). Fine-tune from the champion
checkpoint (-pretrained_weights) with SV_WARMUP=0 to isolate the supervoxel-loss effect at
equal (champion) base quality -> a fast, clean held-out gate signal in hours instead of the
~day the from-scratch run needs to become comparable. Diagnostics, not the deliverable.
"""
from scripts.trainers.nnUNetTrainerSupervoxelTopo import nnUNetTrainerSupervoxelTopo


class nnUNetTrainerSupervoxelTopoProbe(nnUNetTrainerSupervoxelTopo):
    pass
