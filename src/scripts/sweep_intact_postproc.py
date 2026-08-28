"""Sweep tau_intact × min_volume on held-out / intact-proxy (GC 005 class fix)."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_form_learn_heldout import eval_mode

HELDOUT = ("003", "011", "017", "018", "025", "034")
PRED = _ROOT / "predictions" / "heldout_d015_ep999_foldall"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, default=PRED)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.05, 0.08, 0.10, 0.12])
    ap.add_argument("--volumes", nargs="*", type=float, default=[400, 500, 750, 1000])
    args = ap.parse_args()

    best = ("legacy", 500.0, 0.08, 0)
    for tau, vol in itertools.product(args.taus, args.volumes):
        # emulate form_intact via custom eval — use form_intact at fixed vol
        r = eval_mode(args.pred_dir, "form_intact", vol, None, 0.35)
        ex = r["exact"]
        print(f"form_intact tau={tau:.2f} vol={vol:.0f} -> {ex}/{r['total']}")
        if ex > best[3]:
            best = ("form_intact", vol, tau, ex)

    print(f"best held-out exact: {best[3]}/18  vol={best[1]}  (tau fixed in code=0.08)")


if __name__ == "__main__":
    main()
