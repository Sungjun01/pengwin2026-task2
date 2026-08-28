"""Every-50-epoch viz monitor for the D012 w40 + F-CIRL run.

At each 50-epoch milestone it:
  1. snapshots checkpoint_latest.pth -> checkpoint_epoch{N}.pth
  2. predicts 3 fixed held-out cases (011, 017, 025) with that checkpoint
  3. writes a diagnosis PNG  viz/aseg_w40_<fold>/epoch{N}.png  (reuses viz_heldout_diagnosis.py)
  4. writes recovery numbers  viz/aseg_w40_<fold>/recovery_ep{N}.txt

Light: 3 cases on whatever GPU CUDA_VISIBLE_DEVICES points to (launch on a free one).
Robust: each milestone is wrapped in try/except so one failure never kills the loop.

At startup, milestones already passed (training is ahead) are marked done so we
don't snapshot the current checkpoint as if it were the historical one.

Usage (fold defaults to fold_0):
    CUDA_VISIBLE_DEVICES=6 nohup python3 scripts/periodic_viz_aseg_w40.py fold_4 \
        > logs/periodic_viz_aseg_w40_fold_4.log 2>&1 &
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee")
FOLD_NAME = sys.argv[1] if len(sys.argv) > 1 else "fold_0"
FOLD_NUM = FOLD_NAME.removeprefix("fold_")            # "0".."4" or "all"
FOLD = (PROJ.parent / "nnUNet_data/results/Dataset012_PENGWIN_BorderCorePelvic"
        / f"nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres/{FOLD_NAME}")
IN_SRC = PROJ / "predictions/heldout_ep450/in"        # 4-channel held-out inputs
OUTROOT = PROJ / f"viz/aseg_w40_{FOLD_NAME}"
CASES = ["011", "017", "025"]
TARGETS = ["011:RH", "017:RH", "025:LH"]              # known problem cases
MILESTONES = list(range(50, 1000, 50))
EPOCH_RE = re.compile(r"Epoch (\d+)")


def latest_epoch() -> int:
    mx = -1
    for lg in glob.glob(str(FOLD / "training_log_*.txt")):
        try:
            with open(lg, errors="ignore") as f:
                for line in f:
                    m = EPOCH_RE.search(line)
                    if m:
                        mx = max(mx, int(m.group(1)))
        except OSError:
            pass
    return mx


def ensure_input() -> Path:
    d = OUTROOT / "_in3"
    d.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        for ch in range(4):
            src = IN_SRC / f"PENGWIN_{c}_000{ch}.nii.gz"
            dst = d / f"PENGWIN_{c}_000{ch}.nii.gz"
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
    return d


def handle(n: int) -> None:
    chk = FOLD / "checkpoint_latest.pth"
    if not chk.exists():
        return
    snap = FOLD / f"checkpoint_epoch{n}.pth"
    shutil.copy(chk, snap)
    pred = OUTROOT / f"pred_epoch{n}"
    pred.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["nnUNetv2_predict", "-i", str(ensure_input()), "-o", str(pred),
         "-d", "12", "-c", "3d_fullres",
         "-tr", "nnUNetTrainerBorderWeightedAseg_w40", "-f", FOLD_NUM,
         "-chk", snap.name, "-npp", "1", "-nps", "1"],
        check=True, cwd=str(PROJ))
    env = dict(os.environ,
               VIZ_PRED=str(pred), VIZ_CT=str(IN_SRC),
               VIZ_OUT=str(OUTROOT / f"epoch{n:04d}.png"),
               VIZ_TITLE=f"D012 w40+F-CIRL {FOLD_NAME} — epoch {n}")
    subprocess.run(["python3", "scripts/viz_heldout_diagnosis.py", *TARGETS],
                   check=True, cwd=str(PROJ), env=env)
    with open(OUTROOT / f"recovery_ep{n}.txt", "w") as out:
        subprocess.run(["python3", "scripts/measure_border_core_recovery.py",
                        "--pred_dir", str(pred), "--label", f"w40-{FOLD_NAME}-ep{n}",
                        "--min_volume_mm3", "1000"],
                       check=True, cwd=str(PROJ), stdout=out)
    print(f"[viz] {FOLD_NAME} epoch {n}: {OUTROOT}/epoch{n:04d}.png", flush=True)


def main() -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    starting_ep = latest_epoch()
    done: set[int] = {m for m in MILESTONES if m <= starting_ep}
    print(f"[periodic_viz_aseg_w40] watching {FOLD}", flush=True)
    if done:
        print(f"[periodic_viz_aseg_w40] starting at epoch {starting_ep}; "
              f"skipping past milestones {sorted(done)}", flush=True)
    while True:
        ep = latest_epoch()
        for m in [x for x in MILESTONES if x <= ep and x not in done]:
            try:
                handle(m)
            except Exception as e:                       # noqa: BLE001
                print(f"[viz] {FOLD_NAME} epoch {m} FAILED: {e}", flush=True)
            done.add(m)
        if ep >= 999:
            print("[periodic_viz_aseg_w40] training finished, exiting", flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
