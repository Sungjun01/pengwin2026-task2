"""Visualize PENGWIN predictions vs ground truth as multi-panel PNGs.

Usage:
    python scripts/visualize_predictions.py \
        --ct_dir   /path/to/CT/.mha \
        --gt_dir   /path/to/GT/labels \
        --pred_dir /path/to/predictions \
        --out_dir  ./viz \
        [--cases 001 251]    # optional: pick specific case IDs
        [--max_cases 2]      # how many random cases to visualize (default 2)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap


# ---- Color palette ----
ANATOMY_COLORS = [
    (0, 0, 0, 0),         # bg (transparent)
    (0.90, 0.30, 0.30, 0.55),   # sacrum (red)
    (0.30, 0.80, 0.30, 0.55),   # leftHip (green)
    (0.30, 0.50, 0.95, 0.55),   # rightHip (blue)
    (0.95, 0.85, 0.30, 0.55),   # femur (yellow)
]
ANATOMY_CMAP = ListedColormap(ANATOMY_COLORS)

# Instance colors: glasbey-like 50-color palette (cycled)
_INSTANCE_BASE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
INSTANCE_COLORS = [(0, 0, 0, 0)] + [(*c, 0.55) for c in _INSTANCE_BASE]


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()
    return arr, (sz, sy, sx)


def fragment_to_anatomy(lbl: np.ndarray) -> np.ndarray:
    """Map fragment IDs (1-200) to anatomy class (1-4).

    Auto-detects whether input is already anatomy (max <= 4) or fragment IDs.
    If already anatomy, passes through unchanged.
    """
    max_val = int(lbl.max()) if lbl.size else 0
    if max_val <= 4:
        # Already anatomy-coded; just ensure uint8
        return lbl.astype(np.uint8)
    out = np.zeros_like(lbl, dtype=np.uint8)
    out[(lbl >= 1) & (lbl <= 50)] = 1
    out[(lbl >= 51) & (lbl <= 100)] = 2
    out[(lbl >= 101) & (lbl <= 150)] = 3
    out[(lbl >= 151) & (lbl <= 200)] = 4
    return out


def pick_best_slice(mask: np.ndarray, axis: int) -> int:
    """Return slice index with the largest foreground area along `axis`."""
    sums = mask.astype(bool).sum(axis=tuple(a for a in range(mask.ndim) if a != axis))
    if sums.sum() == 0:
        return mask.shape[axis] // 2
    return int(np.argmax(sums))


def overlay_panel(ax, ct_slice: np.ndarray, lbl_slice: np.ndarray, title: str,
                  cmap, is_instance: bool = False) -> None:
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=1500)
    if is_instance:
        # Remap instance IDs to dense 0..N
        unique = sorted(set(np.unique(lbl_slice).tolist()) - {0})
        remap = np.zeros_like(lbl_slice, dtype=np.int32)
        for i, u in enumerate(unique, start=1):
            remap[lbl_slice == u] = i
        max_n = max(len(unique), 1)
        cmap = ListedColormap(INSTANCE_COLORS[: max_n + 1])
        ax.imshow(np.ma.masked_equal(remap, 0), cmap=cmap, vmin=0, vmax=max_n,
                  interpolation="nearest")
    else:
        ax.imshow(np.ma.masked_equal(lbl_slice, 0), cmap=cmap, vmin=0, vmax=4,
                  interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_case(
    case_id: str,
    ct_path: Path,
    gt_path: Path,
    pred_path: Optional[Path],
    out_png: Path,
) -> None:
    ct, _ = load_volume(ct_path)
    gt, _ = load_volume(gt_path)
    pred = None
    if pred_path is not None and pred_path.exists():
        pred, _ = load_volume(pred_path)

    gt_anat = fragment_to_anatomy(gt)
    pred_anat = fragment_to_anatomy(pred) if pred is not None else None

    # Axes: z=0 (axial), 1 (coronal), 2 (sagittal)
    axes_names = ["axial", "coronal", "sagittal"]
    n_rows = 3                # axial, coronal, sagittal
    n_cols = 5                # CT, GT-anat, GT-inst, Pred-anat, Pred-inst
    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(f"Case {case_id} — green=LH, red=sacrum, blue=RH, yellow=femur",
                 fontsize=11)

    for row, axis in enumerate([0, 1, 2]):
        sl = pick_best_slice(gt > 0, axis)
        def _slice(arr): return np.take(arr, sl, axis=axis)
        ct_s = _slice(ct)
        gt_anat_s = _slice(gt_anat)
        gt_inst_s = _slice(gt)
        pred_anat_s = _slice(pred_anat) if pred_anat is not None else None
        pred_inst_s = _slice(pred) if pred is not None else None

        overlay_panel(axarr[row, 0], ct_s, np.zeros_like(ct_s),
                      f"{axes_names[row]} CT (sl={sl})", ANATOMY_CMAP)
        overlay_panel(axarr[row, 1], ct_s, gt_anat_s,
                      "GT anatomy", ANATOMY_CMAP)
        overlay_panel(axarr[row, 2], ct_s, gt_inst_s,
                      "GT instances", ANATOMY_CMAP, is_instance=True)
        if pred_anat_s is not None:
            overlay_panel(axarr[row, 3], ct_s, pred_anat_s,
                          "Pred anatomy", ANATOMY_CMAP)
            overlay_panel(axarr[row, 4], ct_s, pred_inst_s,
                          "Pred instances", ANATOMY_CMAP, is_instance=True)
        else:
            axarr[row, 3].axis("off"); axarr[row, 3].set_title("(no pred)")
            axarr[row, 4].axis("off"); axarr[row, 4].set_title("(no pred)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ct_dir", required=True, type=Path,
                   help="Directory of CT .mha files (subject dirs or flat)")
    p.add_argument("--gt_dir", required=True, type=Path,
                   help="Directory of GT label .mha files (matching basenames)")
    p.add_argument("--pred_dir", type=Path, default=None,
                   help="Directory of prediction label files (optional)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--cases", nargs="*", default=None,
                   help="Specific case IDs to render (e.g. 001 251)")
    p.add_argument("--max_cases", type=int, default=2)
    args = p.parse_args()

    # Find available cases: match files by stem
    gt_files = {p.stem.replace(".nii", ""): p for p in args.gt_dir.glob("*.nii.gz")} or \
               {p.stem: p for p in args.gt_dir.glob("*.mha")}
    ct_files = {}
    # Try direct files (handle nnUNet _0000 channel suffix)
    for ext in (".mha", ".nii.gz"):
        for f in args.ct_dir.rglob(f"*{ext}"):
            stem = f.name.replace(ext, "").replace(".nii", "")
            if stem.endswith("_0000"):
                stem = stem[:-5]
            ct_files.setdefault(stem, f)

    # Resolve case IDs
    if args.cases:
        cases = args.cases
    else:
        cases = sorted(set(gt_files) & set(ct_files))[: args.max_cases]

    if not cases:
        raise SystemExit("No matching cases between CT dir and GT dir.")

    for case in cases:
        ct = ct_files.get(case)
        gt = gt_files.get(case)
        if ct is None or gt is None:
            print(f"[skip] {case}: missing CT or GT")
            continue
        pred = None
        if args.pred_dir is not None:
            for ext in (".nii.gz", ".mha"):
                cand = args.pred_dir / f"{case}{ext}"
                if cand.exists():
                    pred = cand
                    break
        out_png = args.out_dir / f"{case}.png"
        print(f"[viz] {case}  CT={ct.name}  GT={gt.name}  pred={pred.name if pred else 'none'}")
        render_case(case, ct, gt, pred, out_png)

    print(f"\nWrote {len(cases)} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
