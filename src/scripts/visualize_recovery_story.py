"""Visualize the report-37 recovery story: raw core-CC over-seg vs border-adjacency post-proc.

Three cases that tell the whole story:
  001 RH : GT 1  -> raw 7 (1 main bone + 6 satellite blobs) -> postproc 1   (over-seg FIXED)
  003 LH : GT 7  -> raw 9 -> postproc 7                                      (dense, kept right)
  006 sac: GT 9  -> raw 5 -> postproc 5                                      (under-seg, NOT fixable)

Two figures:
  viz/recovery_story_3d.png   - per-fragment 3D meshes, columns GT | A-raw | A-postproc
  viz/recovery_story_ct.png   - CT axial slice + colored instance overlay, same columns
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label
from skimage.measure import marching_cubes

ST = generate_binary_structure(3, 3)
PRED_DIR = "predictions/overfit_recovery/A"
CT_DIR = "/home/schoaq/taehee/nnUNet_data/raw/Dataset099_BC_Overfit/imagesTr"
GT_DIR = "predictions/gt_instance"
ABS_MM = 1000.0
# (case, anatomy name, core_class, border_class, gt_lo, gt_hi)
PANELS = [
    ("001", "rightHip", 5, 6, 101, 150),
    ("003", "leftHip", 3, 4, 51, 100),
    ("006", "sacrum", 1, 2, 1, 50),
]
COLORS = plt.get_cmap("tab20").colors


def bone_window(ct):
    lo, hi = -200.0, 1200.0
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def labeled_gt(ginst, lo, hi, vmm):
    """Return GT instance volume relabeled 1..K over fragments >= ABS_MM in range."""
    out = np.zeros_like(ginst, dtype=np.int32)
    k = 0
    for fid in range(lo, hi + 1):
        m = ginst == fid
        if m.sum() * vmm >= ABS_MM:
            k += 1
            out[m] = k
    return out, k


def labeled_raw(arr, core_cls, vmm):
    """Raw core CC, size filtered, relabeled 1..K."""
    lab, n = cc_label(arr == core_cls, structure=ST)
    out = np.zeros_like(lab, dtype=np.int32)
    k = 0
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() * vmm >= ABS_MM:
            k += 1
            out[m] = k
    return out, k


def labeled_postproc(arr, core_cls, border_cls, vmm):
    """Border-adjacency rule: keep a core CC iff seam-bounded OR largest; relabel kept 1..K."""
    border = arr == border_cls
    lab, n = cc_label(arr == core_cls, structure=ST)
    comps = [(c, int((lab == c).sum())) for c in range(1, n + 1)]
    comps = [(c, s) for c, s in comps if s * vmm >= ABS_MM]
    out = np.zeros_like(lab, dtype=np.int32)
    if not comps:
        return out, 0
    largest = max(comps, key=lambda x: x[1])[0]
    k = 0
    for c, _ in comps:
        m = lab == c
        if (binary_dilation(m, structure=ST, iterations=2) & border).any() or c == largest:
            k += 1
            out[m] = k
    return out, k


def bbox(mask):
    zs, ys, xs = np.where(mask)
    return zs.min(), zs.max() + 1, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def render_3d(ax, vol, k, spacing, title):
    ax.set_title(title, fontsize=11)
    for lid in range(1, k + 1):
        m = vol == lid
        if m.sum() < 30:
            continue
        try:
            verts, faces, _, _ = marching_cubes(m.astype(np.float32), level=0.5,
                                                spacing=spacing, step_size=2)
        except (ValueError, RuntimeError):
            continue
        col = COLORS[(lid - 1) % len(COLORS)]
        mesh = Poly3DCollection(verts[faces], alpha=0.9)
        mesh.set_facecolor(col)
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
    # equal-ish aspect from data extents
    pts = np.argwhere(vol > 0)
    if len(pts):
        mn = pts.min(0) * spacing
        mx = pts.max(0) * spacing
        ax.set_xlim(mn[2], mx[2]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[0], mx[0])
        ax.set_box_aspect((mx[2]-mn[2]+1, mx[1]-mn[1]+1, mx[0]-mn[0]+1))
    ax.view_init(elev=18, azim=-70)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def best_slice(vol):
    return int(np.argmax([(vol[z] > 0).sum() for z in range(vol.shape[0])]))


def main():
    # ---------- gather per-panel volumes ----------
    rows = []
    for cid, name, cc, bc, lo, hi in PANELS:
        pimg = sitk.ReadImage(f"{PRED_DIR}/PENGWIN_{cid}.nii.gz")
        parr = sitk.GetArrayFromImage(pimg)
        sx, sy, sz = pimg.GetSpacing()
        spacing = (sz, sy, sx)  # z,y,x to match array
        vmm = float(sx * sy * sz)
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(f"{GT_DIR}/PENGWIN_{cid}.nii.gz"))
        ct = sitk.GetArrayFromImage(sitk.ReadImage(f"{CT_DIR}/PENGWIN_{cid}_0000.nii.gz")).astype(np.float32)

        gt_v, gt_k = labeled_gt(ginst, lo, hi, vmm)
        raw_v, raw_k = labeled_raw(parr, cc, vmm)
        pp_v, pp_k = labeled_postproc(parr, cc, bc, vmm)

        # shared crop = union bbox of the three label maps
        union = (gt_v > 0) | (raw_v > 0) | (pp_v > 0)
        z0, z1, y0, y1, x0, x1 = bbox(union)
        pad = 4
        z0, y0, x0 = max(0, z0 - pad), max(0, y0 - pad), max(0, x0 - pad)
        z1 = min(union.shape[0], z1 + pad); y1 = min(union.shape[1], y1 + pad); x1 = min(union.shape[2], x1 + pad)
        crop = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        rows.append(dict(cid=cid, name=name, spacing=spacing, ct=ct[crop],
                         gt=(gt_v[crop], gt_k), raw=(raw_v[crop], raw_k), pp=(pp_v[crop], pp_k)))

    # ---------- figure 1: 3D meshes ----------
    fig = plt.figure(figsize=(16, 15))
    for r, row in enumerate(rows):
        for c, (key, lbl) in enumerate([("gt", "GT instances"),
                                        ("raw", "A raw core-CC"),
                                        ("pp", "A + border-adjacency post-proc")]):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
            vol, k = row[key]
            render_3d(ax, vol, k, row["spacing"],
                      f"PENGWIN_{row['cid']} {row['name']}\n{lbl} = {k} piece(s)")
    fig.suptitle("Recovery story (3D): raw core-CC over-segments, border-adjacency post-proc cleans it",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig("viz/recovery_story_3d.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print("wrote viz/recovery_story_3d.png")

    # ---------- figure 2: CT axial overlay ----------
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    for r, row in enumerate(rows):
        z = best_slice((row["raw"][0] > 0) | (row["gt"][0] > 0))
        ctw = bone_window(row["ct"][z])
        for c, (key, lbl) in enumerate([("gt", "GT instances"),
                                        ("raw", "A raw core-CC"),
                                        ("pp", "A + post-proc")]):
            ax = axes[r, c]
            vol, k = row[key]
            ax.imshow(ctw, cmap="gray")
            sl = vol[z]
            ax.imshow(np.ma.masked_where(sl == 0, sl), cmap="tab20", alpha=0.6,
                      vmin=1, vmax=20, interpolation="nearest")
            ax.set_title(f"PENGWIN_{row['cid']} {row['name']}  {lbl} = {k}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Recovery story (CT axial overlay): each color = one fragment",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig("viz/recovery_story_ct.png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    print("wrote viz/recovery_story_ct.png")


if __name__ == "__main__":
    main()
