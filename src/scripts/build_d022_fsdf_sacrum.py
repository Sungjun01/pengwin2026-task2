"""Dataset022 = FS-DF graded 분할 (sacrum 전용). 사크럼 골절면 직접 예측.

D021(femur)의 사크럼 짝. sacrum(1-50) 한 뼈를 골절면까지 거리 4-bin 으로:
  bin1 = seam(<1.5mm), bin2(1.5-3), bin3(3-6), bin4 = core(>6mm).  → class 1-4 + bg.
핵심: 타깃 골절면은 GT 조각 경계(기하)라 *HU 안 보여도 정의됨* → 모델이 003형 dark seam
(-8HU)을 맥락/형상으로 학습. D016 border(HU 구동, dark seam 끊김)의 under-seg 보완.
oracle-채널 함정 회피 위해 입력 = **CT 1채널만** (D006 재사용), offset/side_mask 없음.

입력: D006(SacrumOnly)의 1채널 CT imagesTr 재사용(심링크). 라벨만 FS-DF bin 으로 재생성.

용법: python -m scripts.build_d022_fsdf_sacrum [--limit N]
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
SRC06 = NRAW / "Dataset006_PENGWIN_SacrumOnly"
DST = NRAW / "Dataset022_PENGWIN_FSDF_Sacrum"
LO, HI = 1, 50
THRESH = [1.5, 3.0, 6.0]
LARGE = 100.0


def fsdf_bins(inst, spacing_xyz):
    """sacrum 4bin 라벨 (1-4), bg=0."""
    out = np.zeros(inst.shape, np.uint8)
    samp = spacing_xyz[::-1]  # (z,y,x)
    bm = (inst >= LO) & (inst <= HI)
    if not bm.any():
        return out
    surf = fracture_surface(inst, bm)
    if surf.any():
        d = distance_transform_edt(~surf, sampling=samp).astype(np.float32)
    else:
        d = np.full(inst.shape, LARGE, np.float32)  # 골절 없음 → 전부 core
    binned = np.digitize(d[bm], THRESH) + 1  # 1..4
    flat = out[bm]
    flat[:] = binned
    out[bm] = flat
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name.replace(".nii.gz", "") for p in (SRC06 / "labelsTr").glob("PENGWIN_*.nii.gz"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"[D022] {len(cases)} sacrum cases", flush=True)
    built = 0
    nfrag_hist = {}
    for i, cid in enumerate(cases):
        num = cid.replace("PENGWIN_", "")
        rawlab = RAW / num / "label.mha"
        if not rawlab.exists():
            print(f"  skip {cid}: no raw label"); continue
        gimg = sitk.DICOMOrient(sitk.ReadImage(str(rawlab)), "LPS")
        gt = sitk.GetArrayFromImage(gimg).astype(int)
        gt = np.where((gt >= LO) & (gt <= HI), gt, 0)  # sacrum only
        nfr = len([v for v in np.unique(gt) if v > 0])
        nfrag_hist[nfr] = nfrag_hist.get(nfr, 0) + 1
        lab = fsdf_bins(gt, gimg.GetSpacing())
        lab_img = sitk.GetImageFromArray(lab); lab_img.CopyInformation(gimg)
        sitk.WriteImage(lab_img, str(DST / "labelsTr" / f"{cid}.nii.gz"))
        # CT 1채널 심링크 (D006 재사용)
        src = SRC06 / "imagesTr" / f"{cid}_0000.nii.gz"
        dst = DST / "imagesTr" / f"{cid}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        built += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(cases)}] {cid}", flush=True)

    labels = {"background": 0, "sacrum_bin1": 1, "sacrum_bin2": 2, "sacrum_bin3": 3, "sacrum_bin4": 4}
    ds = {
        "channel_names": {"0": "CT"},
        "labels": labels, "numTraining": built, "file_ending": ".nii.gz",
    }
    (DST / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"[D022] built={built} -> {DST}\nlabels={labels}", flush=True)
    print(f"[D022] sacrum fragment-count histogram (n_frag: n_cases) = {dict(sorted(nfrag_hist.items()))}", flush=True)
    print("D022_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
