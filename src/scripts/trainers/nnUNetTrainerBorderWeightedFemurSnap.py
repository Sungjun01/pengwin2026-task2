"""femur D013 border-core (BorderWeighted) + 50-ep snapshots — best-epoch selection.
Identical loss/recipe to D013 base; only adds periodic checkpoint snapshots so we can gate the
trajectory and pick the instance-best femur epoch (1000 epochs is not always best — overfit risk).
femur = 3-class (bg/core/border), border weight baked into D013 labels.
"""
from __future__ import annotations
import os
from scripts.trainers.nnUNetTrainerBorderWeighted import nnUNetTrainerBorderWeighted


class nnUNetTrainerBorderWeightedFemurSnap(nnUNetTrainerBorderWeighted):
    SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "50"))

    def on_epoch_end(self):
        ep0 = self.current_epoch
        super().on_epoch_end()
        ep = ep0 + 1
        if (self.local_rank == 0 and self.SNAP_EVERY > 0
                and ep % self.SNAP_EVERY == 0 and ep0 != (self.num_epochs - 1)):
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_ep{ep}.pth"))
            self.print_to_log_file(f"[FemurSnap] snapshot -> checkpoint_ep{ep}.pth")
