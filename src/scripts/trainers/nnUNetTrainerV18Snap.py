"""v18 champion recipe (BorderWeightedAseg_w40) + 50-ep snapshots — for best-epoch selection.
Identical loss/recipe to the champion; only adds periodic checkpoint snapshots so we can gate the
trajectory and pick the instance-best epoch (1000 epochs is not always best — overfit risk).
"""
from __future__ import annotations

import os

from scripts.trainers.nnUNetTrainerBorderWeightedAseg_w40 import (
    nnUNetTrainerBorderWeightedAseg_w40,
)


class nnUNetTrainerV18Snap(nnUNetTrainerBorderWeightedAseg_w40):
    SNAP_EVERY = int(os.environ.get("V18_SNAPSHOT_EVERY", "50"))

    def on_epoch_end(self):
        ep0 = self.current_epoch
        super().on_epoch_end()
        ep = ep0 + 1
        if (self.local_rank == 0 and self.SNAP_EVERY > 0
                and ep % self.SNAP_EVERY == 0 and ep0 != (self.num_epochs - 1)):
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_ep{ep}.pth"))
            self.print_to_log_file(f"[V18Snap] snapshot -> checkpoint_ep{ep}.pth")
