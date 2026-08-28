"""FS-DF (D020) 를 실제 public 003 입력에 돌린다 — 첫 정직한 out-of-sample 테스트.

파이프라인(컨테이너와 동일 채널): CT(LPS) → D006 사크럼 → 5채널 → D020(12-class FS-DF)
→ fsdf_decode → instance. 사크럼 fragment 수를 D016(5) / D019(3) 과 비교.
GT 없음 (public) — 구조(특히 사크럼 분할 수)로 판단.

용법: python -m scripts.fsdf_on_real003 --ct "<real003 input .mha>"
"""
from __future__ import annotations
import argparse, os, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import (
    init_predictor, predict_softmax, predict_argmax_multichannel,
    compute_d015_offset_channels, save_temp_channel,
)
from scripts.fsdf_decode_test import fsdf_decode_pelvic

NRES = Path(os.environ.get("nnUNet_results", "/home/schoaq/taehee/nnUNet_data/results"))
D006 = NRES / "Dataset006_PENGWIN_SacrumOnly" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
D020 = NRES / "Dataset020_PENGWIN_FSDF_PredChannels" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="fsdf003_"))

    ct_lps = sitk.DICOMOrient(sitk.ReadImage(args.ct), "LPS")
    spacing_xyz = ct_lps.GetSpacing()
    ct_path = tmp / "case_0000.nii.gz"
    sitk.WriteImage(ct_lps, str(ct_path))
    print(f"[1] CT LPS {ct_lps.GetSize()} spacing {tuple(round(s,3) for s in spacing_xyz)}", flush=True)

    # D006 사크럼
    print("[2] D006 사크럼 추론...", flush=True)
    pa = init_predictor(str(D006))
    probs = predict_softmax(pa, ct_path, tmp / "sac")
    sac_mask = (probs[1].astype(np.float32) > 0.5)
    del pa; torch.cuda.empty_cache()
    print(f"    사크럼 voxels = {int(sac_mask.sum())}", flush=True)

    # 5채널
    offsets = compute_d015_offset_channels(sac_mask, spacing_xyz)
    if offsets is None:
        offsets = np.zeros((3,) + sac_mask.shape, np.float32)
        side = np.zeros(sac_mask.shape, np.float32)
    else:
        side = (offsets[0] > 0).astype(np.float32)
    inputs = [ct_path]
    for i, ch in enumerate(offsets, start=1):
        p = tmp / f"case_{i:04d}.nii.gz"; save_temp_channel(ch, ct_lps, p); inputs.append(p)
    smp = tmp / "case_0004.nii.gz"; save_temp_channel(side, ct_lps, smp); inputs.append(smp)
    print("[3] 5채널 빌드 완료", flush=True)

    # D020 FS-DF (12-class argmax)
    print("[4] D020 FS-DF 추론...", flush=True)
    pb = init_predictor(str(D020))
    sem12 = predict_argmax_multichannel(pb, inputs, tmp / "d020")
    del pb; torch.cuda.empty_cache()

    # FS-DF decode
    inst = fsdf_decode_pelvic(sem12, spacing_xyz, sigma=1.0)
    # 저장 (시각화용)
    out_img = sitk.GetImageFromArray(inst.astype(np.uint16)); out_img.CopyInformation(ct_lps)
    save_path = os.environ.get("FSDF_SAVE", "/tmp/fsdf_real003_inst.nii.gz")
    sitk.WriteImage(out_img, save_path)
    print(f"[saved] FS-DF instance -> {save_path}", flush=True)
    vv = float(np.prod(spacing_xyz))
    u, c = np.unique(inst, return_counts=True)
    anat = {range(1, 51): "sacrum", range(51, 101): "leftHip", range(101, 151): "rightHip"}
    print("\n=== FS-DF (D020) on real public 003 ===")
    for v, cnt in zip(u, c):
        if v == 0:
            continue
        nm = next((n for r, n in anat.items() if v in r), "?")
        print(f"  label {int(v):3d} ({nm:8s}): {cnt*vv/1000:6.1f} cm3")
    for rng, nm in [((1, 50), "sacrum"), ((51, 100), "leftHip"), ((101, 150), "rightHip")]:
        n = len([v for v in u if rng[0] <= v <= rng[1]])
        print(f"  {nm}: {n} fragments")
    pel = [v for v in u if 1 <= v <= 150]
    print(f"  → pelvic 총 {len(pel)} fragments")
    print("  비교: D016(v18,0.952)=11 (사크럼5) / D019(v22,0.841)=9 (사크럼3)")
    print("FSDF_REAL003_DONE")


if __name__ == "__main__":
    main()
