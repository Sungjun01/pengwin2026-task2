"""Monitor nnUNet training and run viz at every 200 epochs.

Usage:
    python3 scripts/periodic_viz.py &

What it does:
  - Tails both training logs for D001 and D002
  - When latest epoch crosses 200, 400, 600, 800:
      * Snapshots `checkpoint_latest.pth` to `checkpoint_epoch{N}.pth`
      * Runs inference on 2 sample cases (pelvic 001, femur 251) for D001
      * Runs inference on per-bone samples for D002 (sacrum + femur)
      * Generates viz PNG in `viz/epoch_{N}/`
  - Stops when both datasets done (or epoch reached 1000).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# ---- Config ----
PROJECT = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee")
NN_RAW = Path("/home/schoaq/taehee/nnUNet_data/raw")
NN_PRE = Path("/home/schoaq/taehee/nnUNet_data/preprocessed")
NN_RES = Path("/home/schoaq/taehee/nnUNet_data/results")

TARGET_EPOCHS = [200, 400, 600, 800]   # 1000 = final, auto-saved by nnUNet

D001_DIR = NN_RES / "Dataset001_PENGWIN_Anatomical/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"
D002_DIR = NN_RES / "Dataset002_PENGWIN_Frac/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"

D001_TEST_INPUT = Path("/tmp/test_input")             # full CTs
D002_TEST_INPUT = Path("/tmp/test_input_d002")        # per-bone masked CTs
TEST_CASES_D001 = ["PENGWIN_001", "PENGWIN_251"]
TEST_CASES_D002 = ["PENGWIN_001_sacrum", "PENGWIN_251_femur"]

CHECK_INTERVAL_SEC = 60
EPOCH_RE = re.compile(r": Epoch (\d+)\s*$")

ENV = {
    **os.environ,
    "nnUNet_raw": str(NN_RAW),
    "nnUNet_preprocessed": str(NN_PRE),
    "nnUNet_results": str(NN_RES),
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "3",   # use GPU 3 for inference (won't disturb training on GPU 1/2)
}

# ---- Helpers ----

def latest_epoch(training_log_glob: list[Path]) -> int:
    """Parse training_log_*.txt and return highest epoch seen."""
    if not training_log_glob:
        return 0
    last = -1
    for p in training_log_glob:
        try:
            with open(p) as f:
                for line in f:
                    m = EPOCH_RE.search(line.rstrip())
                    if m:
                        last = max(last, int(m.group(1)))
        except FileNotFoundError:
            continue
    return last + 1 if last >= 0 else 0   # convert 0-indexed to 1-indexed-ish


def ensure_d002_inputs() -> None:
    """Prepare per-bone CT inputs for D002 sanity inference (one-time)."""
    if D002_TEST_INPUT.exists():
        return
    D002_TEST_INPUT.mkdir(parents=True, exist_ok=True)
    src = PROJECT / "data/Dataset002_PENGWIN_Frac/imagesTr"
    for name in TEST_CASES_D002:
        f = src / f"{name}_0000.nii.gz"
        if f.exists():
            shutil.copy(f, D002_TEST_INPUT / f.name)


def snapshot_checkpoint(dir_: Path, epoch_n: int) -> Path | None:
    """Copy checkpoint_latest.pth to checkpoint_epoch{N}.pth."""
    src = dir_ / "checkpoint_latest.pth"
    if not src.exists():
        return None
    dst = dir_ / f"checkpoint_epoch{epoch_n}.pth"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def run_inference(dataset_id: str, input_dir: Path, output_dir: Path,
                  checkpoint_name: str = "checkpoint_latest.pth") -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", dataset_id, "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-f", "all",
        "-chk", checkpoint_name,
        "--save_probabilities",
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR] {r.stderr[-500:]}", flush=True)
    return r.returncode == 0


def run_viz(ct_dir: Path, gt_dir: Path, pred_dir: Path, out_dir: Path,
            cases: list[str]) -> None:
    cmd = [
        "python3", "scripts/visualize_predictions.py",
        "--ct_dir", str(ct_dir),
        "--gt_dir", str(gt_dir),
        "--pred_dir", str(pred_dir),
        "--out_dir", str(out_dir),
        "--cases", *cases,
    ]
    print(f"[viz] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=ENV, cwd=str(PROJECT))


def handle_milestone(dataset_id: str, dir_: Path, epoch_n: int,
                     input_dir: Path, gt_dir: Path,
                     viz_out: Path, cases: list[str]) -> None:
    # snapshot the checkpoint at this epoch first
    snap = snapshot_checkpoint(dir_, epoch_n)
    if snap is None:
        print(f"[skip] no checkpoint at d{dataset_id} epoch {epoch_n}", flush=True)
        return
    # predict using the snapshotted checkpoint
    pred_dir = PROJECT / f"viz/epoch_{epoch_n}/pred_d{dataset_id}"
    ok = run_inference(dataset_id, input_dir, pred_dir,
                       checkpoint_name=snap.name)
    if not ok:
        return
    run_viz(input_dir, gt_dir, pred_dir, viz_out, cases)


def main():
    print(f"[periodic_viz] starting; targets={TARGET_EPOCHS}", flush=True)
    ensure_d002_inputs()

    done_d001: set[int] = set()
    done_d002: set[int] = set()

    while True:
        d001_logs = sorted(D001_DIR.glob("training_log_*.txt"))
        d002_logs = sorted(D002_DIR.glob("training_log_*.txt"))
        e1 = latest_epoch(d001_logs)
        e2 = latest_epoch(d002_logs)

        print(f"[heartbeat] d001 epoch={e1}  d002 epoch={e2}  "
              f"done_d001={sorted(done_d001)}  done_d002={sorted(done_d002)}",
              flush=True)

        for tgt in TARGET_EPOCHS:
            if e1 > tgt and tgt not in done_d001:
                viz_out = PROJECT / f"viz/epoch_{tgt}"
                handle_milestone("001", D001_DIR, tgt,
                                 D001_TEST_INPUT,
                                 PROJECT / "data/Dataset001_PENGWIN_Anatomical/labelsTr",
                                 viz_out, TEST_CASES_D001)
                done_d001.add(tgt)
            if e2 > tgt and tgt not in done_d002:
                viz_out = PROJECT / f"viz/epoch_{tgt}"
                handle_milestone("002", D002_DIR, tgt,
                                 D002_TEST_INPUT,
                                 PROJECT / "data/Dataset002_PENGWIN_Frac/labelsTr",
                                 viz_out, TEST_CASES_D002)
                done_d002.add(tgt)

        if (set(TARGET_EPOCHS).issubset(done_d001)
                and set(TARGET_EPOCHS).issubset(done_d002)):
            print("[periodic_viz] all milestones reached. exit.", flush=True)
            break

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
