"""004 femur 표면 오차 진단 — D004 femur 마스크 표면이 *어디서* GT와 어긋나나.

004(공개, GT없음)의 HD95 8.67은 *조각분리(decode)가 아니라 표면 분할 품질* 문제(instance 1.0).
femur 학습케이스(GT 있음)로 일반 패턴 파악: 표면 오차가 체계적인가(femoral head/고관절 경계 등)?

각 femur 케이스:
  1. D004 → femur binary 마스크.
  2. GT femur(151-200) binary.
  3. 표면 거리: pred_surf↔gt_surf (양방향 EDT, mm).
  4. HD95 + 고오차(>5mm) 표면 voxel 위치 → CT 위 시각화 + 해부학 영역 추정.

용법: CUDA_VISIBLE_DEVICES=0 python -m scripts.femur_surface_diagnosis
"""
from __future__ import annotations
import sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure, center_of_mass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import init_predictor, predict_softmax

ST = generate_binary_structure(3, 3)
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
RESULTS = Path("/home/schoaq/taehee/nnUNet_data/results")
D004 = RESULTS / "Dataset004_PENGWIN_FemurAnatomy" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
CASES = os.environ.get("PENGWIN_FEMUR_CASES", "251,254,259,279,280,284").split(",")


def surface(mask):
    return mask & ~binary_erosion(mask, ST)


def main():
    pred = init_predictor(str(D004))
    tmp = Path(tempfile.mkdtemp())
    rows = []
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15, 4.2 * len(CASES)))
    if len(CASES) == 1:
        axes = axes[None, :]
    for ri, cid in enumerate(CASES):
        cid = cid.strip()
        ct_img = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "image.mha")), "LPS")
        gt_img = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        gt = sitk.GetArrayFromImage(gt_img)
        sp = ct_img.GetSpacing(); spz = np.array([sp[2], sp[1], sp[0]])
        ct_path = tmp / "ct_0000.nii.gz"; sitk.WriteImage(ct_img, str(ct_path))
        probs = predict_softmax(pred, ct_path, tmp / "d004")
        fem_pred = (probs[1].astype(np.float32) > 0.5)
        fem_gt = (gt >= 151) & (gt <= 200)
        if not fem_pred.any() or not fem_gt.any():
            continue
        ps, gs = surface(fem_pred), surface(fem_gt)
        # pred surface → 가장 가까운 gt surface 거리 (mm)
        dt_to_gt = distance_transform_edt(~gs, sampling=spz)
        dist_ps = dt_to_gt[ps]
        dt_to_pred = distance_transform_edt(~ps, sampling=spz)
        dist_gs = dt_to_pred[gs]
        alld = np.concatenate([dist_ps, dist_gs])
        hd95 = float(np.percentile(alld, 95)); assd = float(alld.mean())
        # 고오차 표면 voxel (>5mm) 위치
        err_mask = np.zeros_like(fem_pred)
        err_mask[ps] = dist_ps > 5.0
        err_n = int(err_mask.sum())
        loc = "없음"
        if err_n > 0:
            cz, cy, cx = center_of_mass(err_mask)
            zrel = cz / fem_pred.shape[0]
            loc = f"z={cz:.0f}({'상부/고관절' if zrel<0.4 else ('중간' if zrel<0.7 else '하부/shaft')}), n={err_n}"
        rows.append((cid, hd95, assd, err_n, loc))
        print(f"{cid}: HD95={hd95:.2f} ASSD={assd:.2f} 고오차(>5mm)voxel={err_n} 위치={loc}", flush=True)
        # 시각화: 고오차 voxel이 가장 많은 z 슬라이스
        zc = err_mask.sum(axis=(1, 2)).argmax() if err_n > 0 else fem_pred.sum(axis=(1, 2)).argmax()
        for ci, (lab, ov) in enumerate([("CT+GT femur", fem_gt[zc]), ("CT+D004 femur", fem_pred[zc]),
                                        ("pred surf 오차(mm)", None)]):
            ax = axes[ri, ci]
            ax.imshow(ct[zc], cmap="gray", vmin=-200, vmax=1100)
            if ci < 2:
                ax.imshow(np.ma.masked_where(~ov, ov), cmap="autumn", alpha=0.4)
            else:
                em = np.full(ps[zc].shape, np.nan); em[ps[zc]] = dt_to_gt[zc][ps[zc]]
                im = ax.imshow(np.ma.masked_invalid(em), cmap="jet", vmin=0, vmax=10)
            ax.set_title(f"{cid} {lab}" + (f"\nHD95={hd95:.1f}" if ci == 2 else ""), fontsize=9); ax.axis("off")
    plt.suptitle("Femur surface error diagnosis: D004 mask surface vs GT (jet=distance mm, red=>10mm). "
                 "Where is the ~8.67mm error?", fontsize=12)
    plt.tight_layout(); plt.savefig("figs/femur_surface_diag.png", dpi=110, bbox_inches="tight"); plt.close()
    print("\nsaved figs/femur_surface_diag.png")
    print("\n=== 요약 ===")
    print(f"{'case':6s} {'HD95':>6s} {'ASSD':>6s} {'고오차voxel':>10s}  위치")
    for cid, hd, ad, en, loc in rows:
        print(f"{cid:6s} {hd:6.2f} {ad:6.2f} {en:10d}  {loc}")
    print("FEMUR_DIAG_DONE")


if __name__ == "__main__":
    main()
