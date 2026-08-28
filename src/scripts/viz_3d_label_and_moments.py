"""PENGWIN case 의 GT label 4-class anatomy + sacrum centroid + PCA axes 3D 시각화.

skimage.measure.marching_cubes 로 per-class surface mesh 추출 후 plotly Mesh3d
(interactive HTML) + matplotlib Poly3DCollection (static PNG) 둘 다 출력.

Visualization 요소.
    sacrum   : red surface
    leftHip  : green surface
    rightHip : blue surface
    sacrum centroid : black ★ marker
    PCA principal axes : 3 arrows from centroid (1st = red, 2nd = green, 3rd = blue dashed)

용법.
    python scripts/viz_3d_label_and_moments.py \\
        --cases PENGWIN_001 \\
        --out_dir docs/figures/spatial_moments_3d
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


CLASS_COLORS = {
    1: ("sacrum", "#c0392b"),     # red
    2: ("leftHip", "#27ae60"),    # green
    3: ("rightHip", "#2980b9"),   # blue
}


def compute_sacrum_moments(label_arr: np.ndarray, spacing: tuple[float, float, float]):
    """Sacrum (class 1) 의 mass center + PCA eigendecomposition in physical mm coordinate.

    Returns:
        centroid_mm   : (3,) in (z, y, x) order
        eigenvalues   : (3,)
        eigenvectors  : (3, 3) — columns are eigenvectors in (z, y, x) order
    """
    coords = np.argwhere(label_arr == 1)
    if coords.shape[0] < 10:
        sys.exit("sacrum voxel count too low for PCA")
    spacing_arr = np.array(spacing[::-1])  # SimpleITK GetSpacing 은 (x, y, z), 우리 array 는 (z, y, x)
    coords_mm = coords * spacing_arr
    centroid = coords_mm.mean(axis=0)
    cov = np.cov(coords_mm.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return centroid, eigenvalues, eigenvectors


def extract_meshes(label_arr: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict:
    """Per-class marching cubes mesh in physical mm coordinate.

    Returns:
        {class_id: (vertices, faces)} where vertices are in mm (z, y, x).
    """
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cls in (1, 2, 3):
        mask = (label_arr == cls).astype(np.uint8)
        if mask.sum() < 100:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing_zyx)
        except (RuntimeError, ValueError) as e:
            print(f"  [warn] marching_cubes failed for class {cls}: {e}")
            continue
        meshes[cls] = (verts, faces)
    return meshes


def render_plotly_html(meshes, centroid, eigenvalues, eigenvectors, out_path: Path, title: str):
    if go is None:
        return None
    fig = go.Figure()

    for cls, (verts, faces) in meshes.items():
        name, color = CLASS_COLORS[cls]
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],  # plotly axes: x, y, z
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.55, name=name, showscale=False,
            lighting=dict(ambient=0.45, diffuse=0.7, specular=0.2),
        ))

    # sacrum centroid as black star
    fig.add_trace(go.Scatter3d(
        x=[centroid[2]], y=[centroid[1]], z=[centroid[0]],
        mode="markers+text",
        marker=dict(size=8, color="black", symbol="cross"),
        text=["sacrum centroid"], textposition="top center",
        name="sacrum centroid",
    ))

    # PCA principal axes (3) — eigenvalue 순으로 정렬, 큰 게 마지막 (ascending)
    axis_scales = np.sqrt(np.maximum(eigenvalues, 0))
    axis_colors = ["#34495e", "#7f8c8d", "#bdc3c7"]
    axis_names = ["PCA axis 3 (longest)", "PCA axis 2", "PCA axis 1 (shortest)"]
    # eigh returns ascending eigenvalues; reverse for visual labeling
    order = list(range(3))[::-1]
    for rank, idx in enumerate(order):
        v = eigenvectors[:, idx] * axis_scales[idx] * 2.5  # scale factor for visibility
        # 양쪽으로 ±v
        for sign, suffix in [(+1, "+"), (-1, "-")]:
            tip = centroid + sign * v
            fig.add_trace(go.Scatter3d(
                x=[centroid[2], tip[2]],
                y=[centroid[1], tip[1]],
                z=[centroid[0], tip[0]],
                mode="lines",
                line=dict(color=axis_colors[rank], width=6),
                name=f"{axis_names[rank]} ({suffix})" if sign == 1 else None,
                showlegend=(sign == 1),
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


def render_matplotlib_png(meshes, centroid, eigenvalues, eigenvectors, out_path: Path, title: str):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for cls, (verts, faces) in meshes.items():
        name, color = CLASS_COLORS[cls]
        # subsample faces if too many
        n_faces = len(faces)
        step = max(1, n_faces // 8000)
        triangles = verts[faces[::step]]
        poly = Poly3DCollection(
            triangles[:, :, [2, 1, 0]],  # x, y, z order
            alpha=0.5, facecolor=color, edgecolor="none",
        )
        ax.add_collection3d(poly)
        ax.plot([], [], [], color=color, label=name)

    # centroid + PCA axes
    ax.scatter([centroid[2]], [centroid[1]], [centroid[0]],
               color="black", marker="*", s=180, label="sacrum centroid", zorder=10)
    axis_scales = np.sqrt(np.maximum(eigenvalues, 0))
    axis_colors = ["#34495e", "#7f8c8d", "#bdc3c7"]
    for rank, idx in enumerate([2, 1, 0]):
        v = eigenvectors[:, idx] * axis_scales[idx] * 2.5
        for sign in (+1, -1):
            tip = centroid + sign * v
            ax.plot(
                [centroid[2], tip[2]],
                [centroid[1], tip[1]],
                [centroid[0], tip[0]],
                color=axis_colors[rank], linewidth=2, alpha=0.85,
            )

    # combined bbox for axes
    all_verts = np.concatenate([v for v, _ in meshes.values()], axis=0) if meshes else np.array([centroid])
    mins = all_verts.min(axis=0)
    maxs = all_verts.max(axis=0)
    ax.set_xlim(mins[2], maxs[2])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[0], maxs[0])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.view_init(elev=20, azim=-60)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_case(case_id: str, label_dir: Path, out_dir: Path) -> None:
    label_path = label_dir / f"{case_id}.nii.gz"
    if not label_path.exists():
        print(f"  [skip] {case_id}: label 없음 ({label_path})")
        return

    img = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(img)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    centroid, eigvals, eigvecs = compute_sacrum_moments(label_arr, spacing_xyz)
    print(f"  [case] {case_id} — centroid (z,y,x) = ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm")
    print(f"         eigenvalues (ascending) = {eigvals.round(2).tolist()}")

    meshes = extract_meshes(label_arr, spacing_zyx)
    if not meshes:
        print(f"  [skip] {case_id}: no mesh extracted")
        return

    title = f"PENGWIN {case_id} — GT anatomy + sacrum centroid + PCA axes"
    png_path = render_matplotlib_png(meshes, centroid, eigvals, eigvecs,
                                     out_dir / f"{case_id}_3d.png", title)
    print(f"  [saved] {png_path}")
    html_path = render_plotly_html(meshes, centroid, eigvals, eigvecs,
                                   out_dir / f"{case_id}_3d.html", title)
    if html_path:
        print(f"  [saved] {html_path}  (interactive — open in browser, drag to rotate)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["PENGWIN_001"])
    parser.add_argument("--label_dir", type=Path, default=None,
                        help="default = $nnUNet_raw/Dataset003_PENGWIN_PelvicAnatomy/labelsTr")
    parser.add_argument("--out_dir", type=Path, default=Path("docs/figures/spatial_moments_3d"))
    args = parser.parse_args()

    if args.label_dir is None:
        nnraw = Path(env("nnUNet_raw"))
        args.label_dir = nnraw / "Dataset003_PENGWIN_PelvicAnatomy" / "labelsTr"

    print(f"[setup] label_dir = {args.label_dir}")
    print(f"[setup] cases = {args.cases}")
    print(f"[setup] out = {args.out_dir}")

    for case_id in args.cases:
        render_case(case_id, args.label_dir, args.out_dir)

    print(f"\nDONE — 3D figures at {args.out_dir}")


if __name__ == "__main__":
    main()
