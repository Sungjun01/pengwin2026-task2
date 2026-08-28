"""Dataset018 = D015 와 동일하되 offset/side_mask 채널을 GT 사크럼이 아니라
D006 *예측* 사크럼으로 생성. train/inference 분포 일치 (oracle-channel 함정 제거).

CT(ch0) + border-core 라벨은 D015 와 동일 공식 (GT label). 채널 1-4 만 예측 사크럼.

용법: CUDA_VISIBLE_DEVICES=N python -m scripts.build_d018_predicted_channels [--limit N]
"""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_d015_oriented_sidemask import (
    case_ids, compute_offsets, save_volume, RAW, LABELS,
)
from scripts.border_core_conversion import convert_to_border_core, PELVIC_ANATOMY
from scripts.inference_v7_d015 import init_predictor, predict_softmax

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
RESULTS = Path(os.environ.get("nnUNet_results", "/home/schoaq/taehee/nnUNet_data/results"))
DST = NRAW / "Dataset018_PENGWIN_BorderCore_PredChannels"
D006_DIR = RESULTS / "Dataset006_PENGWIN_SacrumOnly" / "nnUNetTrainer__nnUNetPlans__3d_fullres"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    print(f"[D018] init D006 sacrum predictor: {D006_DIR}", flush=True)
    predictor = init_predictor(str(D006_DIR), tile_step_size=1.0)

    cids = case_ids()
    if args.limit:
        cids = cids[: args.limit]
    print(f"[D018] {len(cids)} cases (predicted-sacrum channels, LPS)", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="d018_"))
    built, skipped = 0, []
    for i, cid in enumerate(cids):
        num = cid.replace("PENGWIN_", "")
        src_img, src_lab = RAW / num / "image.mha", RAW / num / "label.mha"
        if not src_img.exists() or not src_lab.exists():
            skipped.append(f"{cid}:missing"); continue
        img = sitk.DICOMOrient(sitk.ReadImage(str(src_img)), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(str(src_lab)), "LPS")
        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        labarr = sitk.GetArrayFromImage(lab)
        spacing = img.GetSpacing()

        # --- 예측 사크럼 (D006) on LPS CT ---
        ct_path = tmp / "ct_0000.nii.gz"
        sitk.WriteImage(img, str(ct_path))
        probs = predict_softmax(predictor, ct_path, tmp / "sac")
        sac_mask = (probs[1].astype(np.float32) > 0.5).astype(np.float32)
        if sac_mask.sum() < 1e-3:
            skipped.append(f"{cid}:no-pred-sacrum"); continue

        nx, ny, nz = compute_offsets(sac_mask, spacing)
        side = (nx > 0).astype(np.float32)
        bc = convert_to_border_core(labarr, PELVIC_ANATOMY, t=2)

        save_volume(ct, img, str(DST / "imagesTr" / f"{cid}_0000.nii.gz"))
        save_volume(nx, img, str(DST / "imagesTr" / f"{cid}_0001.nii.gz"))
        save_volume(ny, img, str(DST / "imagesTr" / f"{cid}_0002.nii.gz"))
        save_volume(nz, img, str(DST / "imagesTr" / f"{cid}_0003.nii.gz"))
        save_volume(side, img, str(DST / "imagesTr" / f"{cid}_0004.nii.gz"))
        save_volume(bc, img, str(DST / "labelsTr" / f"{cid}.nii.gz"), dtype=np.uint8)
        built += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(cids)}] {cid} built", flush=True)

    ds = {
        "channel_names": {"0": "CT", "1": "offset1", "2": "offset2",
                          "3": "offset3", "4": "side_mask"},
        "labels": LABELS,
        "numTraining": built,
        "file_ending": ".nii.gz",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D018] built={built} skipped={len(skipped)} -> {DST}", flush=True)
    if skipped:
        print("  skipped:", skipped[:10], flush=True)
    print("D018_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
