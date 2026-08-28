"""femur TTA 시간·토폴로지 sanity — 빠른 도박 컨테이너가 T4 예산에 맞는지 확인.

D004 femur 추론을 (1) TTA off, (2) TTA full(model-default axes), (3) TTA 1축
세 가지로 돌려 wall-clock + femur binary 변화(voxel %, CC count)를 비교.
정확도 검증 아님 — *예산 내 실행 + 토폴로지 보존* 확인용.

용법: CUDA_VISIBLE_DEVICES=0 python3 -m scripts.femur_tta_timing 254
"""
from __future__ import annotations
import sys, time, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import init_predictor, predict_softmax

ST = generate_binary_structure(3, 3)
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
D004 = Path("/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy/nnUNetTrainer__nnUNetPlans__3d_fullres")
MIN_VOL = 500.0


def femur_bin_and_cc(probs, vv):
    b = (probs.argmax(axis=0) == 1)
    labels, n = cc_label(b, structure=ST)
    kept = sum(1 for c in range(1, n + 1) if (labels == c).sum() * vv >= MIN_VOL)
    return b, kept


def run(cfg_name, use_mirroring, axes, ct_path, tmp, vv):
    p = init_predictor(str(D004), use_mirroring=use_mirroring)
    if use_mirroring and axes:
        p.allowed_mirroring_axes = tuple(axes)
    t = time.time()
    probs = predict_softmax(p, ct_path, tmp / cfg_name)
    dt = time.time() - t
    b, kept = femur_bin_and_cc(probs, vv)
    return dt, b, kept, getattr(p, "allowed_mirroring_axes", None)


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "254"
    ct_img = sitk.ReadImage(str(RAW / cid / "image.mha"))
    sx, sy, sz = ct_img.GetSpacing(); vv = float(sx * sy * sz)
    tmp = Path(tempfile.mkdtemp(prefix="ftta_"))
    ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct_img, str(ct_path))
    print(f"case {cid}  spacing={ct_img.GetSpacing()}  shape={ct_img.GetSize()}", flush=True)

    dt0, b0, k0, _ = run("off", False, None, ct_path, tmp, vv)
    print(f"[off ] time={dt0:6.1f}s  femur_vox={int(b0.sum()):>8d}  CC(>500mm3)={k0}", flush=True)

    dt1, b1, k1, ax1 = run("full", True, None, ct_path, tmp, vv)
    chg1 = 100.0 * (b0 ^ b1).sum() / max(int(b0.sum()), 1)
    print(f"[full] time={dt1:6.1f}s  ({dt1/dt0:.1f}x)  axes={ax1}  CC={k1}  voxelΔ={chg1:.2f}%", flush=True)

    dt2, b2, k2, ax2 = run("ax2", True, (2,), ct_path, tmp, vv)
    chg2 = 100.0 * (b0 ^ b2).sum() / max(int(b0.sum()), 1)
    print(f"[ax2 ] time={dt2:6.1f}s  ({dt2/dt0:.1f}x)  axes={ax2}  CC={k2}  voxelΔ={chg2:.2f}%", flush=True)

    print("\n=== 판정 ===")
    print(f"  off  {dt0:5.1f}s | full {dt1:5.1f}s | ax2 {dt2:5.1f}s")
    print(f"  토폴로지 보존(CC): off={k0} full={k1} ax2={k2}  → {'OK' if k0==k1==k2 else 'CHANGED!'}")
    print("FEMUR_TTA_TIMING_DONE")


if __name__ == "__main__":
    main()
