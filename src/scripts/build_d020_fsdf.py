"""Dataset020 = FS-DF graded 분할 (pelvic). 새 방법론 학습용.

border-core(core/border 2단계)를 → 골절면까지 거리 4-bin 으로 일반화:
  bin1 = seam(<1.5mm), bin2(1.5-3), bin3(3-6), bin4 = core(>6mm).
anatomy(sac/LH/RH) × 4bin = 12 class + bg.

입력: D019 의 예측-채널 imagesTr 재사용(심링크). 라벨만 FS-DF bin 으로 재생성.
decode(추론): 예측 bin → 연속 거리장 복원 → core CC marker watershed.

용법: python -m scripts.build_d020_fsdf [--limit N]
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fsdf_experiment import fracture_surface

NRAW = Path(os.environ.get("nnUNet_raw", "/home/schoaq/taehee/nnUNet_data/raw"))
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SRC19 = NRAW / "Dataset019_PENGWIN_BorderCore_PredChannels_Retrain"
DST = NRAW / "Dataset020_PENGWIN_FSDF_PredChannels"
ANAT = {"sacrum": (1, 50, 0), "leftHip": (51, 100, 4), "rightHip": (101, 150, 8)}  # (lo,hi,class_base)
THRESH = [1.5, 3.0, 6.0]  # bin 경계 (mm) → bin 1/2/3/4
LARGE = 100.0


def fsdf_bins(inst, spacing_xyz):
    """anatomy×4bin 라벨 (1-12), bg=0."""
    out = np.zeros(inst.shape, np.uint8)
    samp = spacing_xyz[::-1]  # (z,y,x)
    for nm, (lo, hi, base) in ANAT.items():
        bm = (inst >= lo) & (inst <= hi)
        if not bm.any():
            continue
        surf = fracture_surface(inst, bm)
        if surf.any():
            d = distance_transform_edt(~surf, sampling=samp).astype(np.float32)
        else:
            d = np.full(inst.shape, LARGE, np.float32)  # 골절 없음 → 전부 core
        db = d[bm]
        binned = np.digitize(db, THRESH) + 1  # 1..4
        flat = out[bm]
        flat[:] = base + binned
        out[bm] = flat
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name.replace(".nii.gz", "") for p in (SRC19 / "labelsTr").glob("PENGWIN_*.nii.gz"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"[D020] {len(cases)} cases", flush=True)
    built = 0
    for i, cid in enumerate(cases):
        num = cid.replace("PENGWIN_", "")
        rawlab = RAW / num / "label.mha"
        if not rawlab.exists():
            print(f"  skip {cid}: no raw label"); continue
        gimg = sitk.DICOMOrient(sitk.ReadImage(str(rawlab)), "LPS")
        gt = sitk.GetArrayFromImage(gimg).astype(int)
        gt = np.where((gt >= 1) & (gt <= 150), gt, 0)  # pelvic only
        lab = fsdf_bins(gt, gimg.GetSpacing())
        lab_img = sitk.GetImageFromArray(lab); lab_img.CopyInformation(gimg)
        sitk.WriteImage(lab_img, str(DST / "labelsTr" / f"{cid}.nii.gz"))
        # 이미지 5채널 심링크 (D019 재사용)
        for ch in range(5):
            src = SRC19 / "imagesTr" / f"{cid}_{ch:04d}.nii.gz"
            dst = DST / "imagesTr" / f"{cid}_{ch:04d}.nii.gz"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(cases)}] {cid}", flush=True)

    labels = {"background": 0}
    for nm, (lo, hi, base) in ANAT.items():
        for b in range(1, 5):
            labels[f"{nm}_bin{b}"] = base + b
    ds = {
        "channel_names": {"0": "CT", "1": "offset1", "2": "offset2", "3": "offset3", "4": "side_mask"},
        "labels": labels, "numTraining": built, "file_ending": ".nii.gz",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D020] built={built} -> {DST}\nlabels={labels}\nD020_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
