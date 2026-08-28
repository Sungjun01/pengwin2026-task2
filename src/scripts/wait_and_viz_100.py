"""Wait until D003 + D004 both reach epoch 100, snapshot checkpoints, run inference, viz.

Background watcher — runs in nohup. Logs to /tmp/wait_and_viz_100.log.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee")
RES_ROOT = Path("/home/schoaq/taehee/nnUNet_data/results")

D003_DIR = RES_ROOT / "Dataset003_PENGWIN_PelvicAnatomy/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"
D004_DIR = RES_ROOT / "Dataset004_PENGWIN_FemurAnatomy/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all"

# Test inputs: pelvic case for D003, femur case for D004
TEST_PELVIC = Path("/tmp/test_pelvic")
TEST_FEMUR = Path("/tmp/test_femur")
PRED_D003 = PROJECT / "viz/exp2_specialist_d003_d004/ep100/pred_d003"
PRED_D004 = PROJECT / "viz/exp2_specialist_d003_d004/ep100/pred_d004"
VIZ_OUT = PROJECT / "viz/exp2_specialist_d003_d004/ep100"

# D003: pelvic. Use PENGWIN_001 (pelvic case from Dataset001)
# D004: femur. Use PENGWIN_251 (femur case from Dataset001)
PELVIC_TEST_CASE = "PENGWIN_001"
FEMUR_TEST_CASE = "PENGWIN_251"
TARGET_EPOCH = 100
CHECK_INTERVAL_SEC = 60

EPOCH_RE = re.compile(r": Epoch (\d+)\s*$")

ENV = {
    **os.environ,
    "nnUNet_raw": "/home/schoaq/taehee/nnUNet_data/raw",
    "nnUNet_preprocessed": "/home/schoaq/taehee/nnUNet_data/preprocessed",
    "nnUNet_results": str(RES_ROOT),
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "3",   # free GPU
}


def latest_epoch(dir_: Path) -> int:
    logs = sorted(dir_.glob("training_log_*.txt"))
    if not logs:
        return 0
    last = -1
    for p in logs:
        try:
            for line in open(p):
                m = EPOCH_RE.search(line.rstrip())
                if m:
                    last = max(last, int(m.group(1)))
        except FileNotFoundError:
            continue
    return last + 1 if last >= 0 else 0


def prepare_inputs() -> None:
    TEST_PELVIC.mkdir(parents=True, exist_ok=True)
    TEST_FEMUR.mkdir(parents=True, exist_ok=True)
    src = PROJECT / "data/Dataset001_PENGWIN_Anatomical/imagesTr"
    pelvic_src = src / f"{PELVIC_TEST_CASE}_0000.nii.gz"
    femur_src = src / f"{FEMUR_TEST_CASE}_0000.nii.gz"
    pelvic_dst = TEST_PELVIC / f"{PELVIC_TEST_CASE}_0000.nii.gz"
    femur_dst = TEST_FEMUR / f"{FEMUR_TEST_CASE}_0000.nii.gz"
    if not pelvic_dst.exists():
        shutil.copy(pelvic_src, pelvic_dst)
    if not femur_dst.exists():
        shutil.copy(femur_src, femur_dst)


def snapshot_checkpoint(dir_: Path, epoch_n: int) -> Path | None:
    src = dir_ / "checkpoint_latest.pth"
    if not src.exists():
        return None
    dst = dir_ / f"checkpoint_ep{epoch_n}.pth"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def run_predict(dataset_id: str, input_dir: Path, output_dir: Path,
                checkpoint_name: str) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", dataset_id, "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-f", "all",
        "-chk", checkpoint_name,
    ]
    print(f"[predict d{dataset_id}] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR predict d{dataset_id}] {r.stderr[-800:]}", flush=True)
        return False
    return True


def remap_d004_pred(pred_dir: Path) -> None:
    """D004 output has value 1 = femur. Remap to 4 so viz cmap uses yellow."""
    for f in pred_dir.glob("*.nii.gz"):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        new_arr = np.where(arr == 1, 4, 0).astype(np.uint8)
        new_img = sitk.GetImageFromArray(new_arr)
        new_img.CopyInformation(img)
        sitk.WriteImage(new_img, str(f), useCompression=True)


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


def main():
    print(f"[wait_and_viz_100] starting. target epoch = {TARGET_EPOCH}", flush=True)
    prepare_inputs()

    while True:
        e3 = latest_epoch(D003_DIR)
        e4 = latest_epoch(D004_DIR)
        print(f"[heartbeat] d003 epoch={e3}  d004 epoch={e4}", flush=True)
        if e3 >= TARGET_EPOCH and e4 >= TARGET_EPOCH:
            break
        time.sleep(CHECK_INTERVAL_SEC)

    print("[ready] both reached 100 epochs. snapshot + predict + viz.", flush=True)

    snap3 = snapshot_checkpoint(D003_DIR, TARGET_EPOCH)
    snap4 = snapshot_checkpoint(D004_DIR, TARGET_EPOCH)
    print(f"[snap] d003={snap3}, d004={snap4}", flush=True)

    if snap3:
        run_predict("003", TEST_PELVIC, PRED_D003, snap3.name)
        run_viz(
            ct_dir=TEST_PELVIC,
            gt_dir=PROJECT / "data/Dataset003_PENGWIN_PelvicAnatomy/labelsTr",
            pred_dir=PRED_D003,
            out_dir=VIZ_OUT,
            cases=[PELVIC_TEST_CASE],
        )
    if snap4:
        run_predict("004", TEST_FEMUR, PRED_D004, snap4.name)
        # Remap pred from {0,1} to {0,4} so viz cmap shows yellow for femur
        remap_d004_pred(PRED_D004)
        run_viz(
            ct_dir=TEST_FEMUR,
            gt_dir=PROJECT / "data/Dataset004_PENGWIN_FemurAnatomy/labelsTr",
            pred_dir=PRED_D004,
            out_dir=VIZ_OUT,
            cases=[FEMUR_TEST_CASE],
        )

    print("[done] viz/exp2_specialist_d003_d004/ep100/ populated.", flush=True)


if __name__ == "__main__":
    main()
