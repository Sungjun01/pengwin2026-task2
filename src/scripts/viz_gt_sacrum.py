"""사크럼 골절 케이스 GT 시각화 — CT + GT + FS-DF (+선택 D016) 사크럼 비교.

사크럼은 정중선에서 세로로 뻗으므로 coronal/sagittal 슬라이스가 골절선을 잘 보여줌.
GT 사크럼 조각을 색으로 → FS-DF 가 같은 골절선을 잡는지 눈으로 검증.

용법: python -m scripts.viz_gt_sacrum --case 097 [--fsdf <fsdf예측12class>] [--d016 <border예측>]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fsdf_decode_test import fsdf_decode_pelvic
from scripts.form_learn_postproc import assemble_pelvic_instances_mode

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
GTD = "predictions/gt_instance"


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac_relabel(slice2d):
    out = np.zeros_like(slice2d)
    for i, v in enumerate([v for v in np.unique(slice2d) if 1 <= v <= 50], 1):
        out[slice2d == v] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--fsdf", default=None, help="D020 12-class 예측 .nii.gz")
    ap.add_argument("--d016", default=None, help="D016 7-class 예측 .nii.gz")
    ap.add_argument("--axis", default="coronal", choices=["axial", "coronal", "sagittal"])
    args = ap.parse_args()
    c = args.case

    ctimg, ct = lps(Path(RAW) / c / "image.mha")
    _, gt = lps(Path(GTD) / f"PENGWIN_{c}.nii.gz")
    sp = ctimg.GetSpacing()

    panels = [("CT only", None), ("GT (ground truth)", gt)]
    if args.fsdf:
        _, sem = lps(args.fsdf)
        fsdf_inst = fsdf_decode_pelvic(sem, sp, sigma=1.0)
        panels.append(("FS-DF (D020)", fsdf_inst))
    if args.d016:
        d16img, sem16 = lps(args.d016)
        vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        d16_inst = assemble_pelvic_instances_mode(sem16, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        panels.append(("D016 (v18)", d16_inst))

    # GT 사크럼이 가장 많이 보이는 슬라이스 (선택 축)
    gsac = np.where((gt >= 1) & (gt <= 50), gt, 0)
    if args.axis == "axial":
        cnt = [(len(np.unique(gsac[z])[np.unique(gsac[z]) > 0]), z) for z in range(gsac.shape[0])]
        idx = max(cnt)[1]; sl = lambda a: a[idx]
        aspect = sp[1] / sp[0]
    elif args.axis == "coronal":
        cnt = [(len(np.unique(gsac[:, y, :])[np.unique(gsac[:, y, :]) > 0]), y) for y in range(gsac.shape[1])]
        idx = max(cnt)[1]; sl = lambda a: a[:, idx, :]
        aspect = sp[2] / sp[0]
    else:  # sagittal
        cnt = [(len(np.unique(gsac[:, :, x])[np.unique(gsac[:, :, x]) > 0]), x) for x in range(gsac.shape[2])]
        idx = max(cnt)[1]; sl = lambda a: a[:, :, idx]
        aspect = sp[2] / sp[1]
    print(f"{args.axis} slice idx={idx}, GT 사크럼 {max(cnt)[0]}개 보임")

    cts = sl(ct)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6))
    for i, (name, m) in enumerate(panels):
        axes[i].imshow(cts, cmap="gray", vmin=-200, vmax=900, origin="lower", aspect=aspect)
        if m is not None:
            rl = sac_relabel(sl(np.where((m >= 1) & (m <= 50), m, 0)))
            nfull = len([v for v in np.unique(m) if 1 <= v <= 50])
            axes[i].imshow(np.ma.masked_where(rl == 0, rl), cmap="tab10", alpha=0.75,
                           vmin=1, vmax=10, origin="lower", aspect=aspect)
            axes[i].set_title(f"{name}\nsacrum {nfull} fragments")
        else:
            axes[i].set_title(name)
        axes[i].axis("off")
    plt.suptitle(f"Training case {c} — SACRUM fracture (GT-grounded, {args.axis})", fontsize=14)
    plt.tight_layout()
    out = f"figs/gt_sacrum_{c}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
