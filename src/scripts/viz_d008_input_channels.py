"""Dataset008 의 8 channel input 을 한 case 의 같은 axial slice 에서 multi-panel 로 시각화.

paper §4 method figure 후보. *각 voxel 이 sacrum 의 anatomical coordinate frame 에서
어디인지* 가 입력 단에 명시적으로 broadcast 된다는 mechanism 을 직접 그림.

Channel 구성.
    0 : CT (HU window)
    1 : P_sacrum (sigmoid 확률맵)
    2 : rel_z (voxel z - centroid z, case dim normalized)
    3 : rel_y (voxel y - centroid y)
    4 : rel_x (voxel x - centroid x)
    5 : pca_1 (sacrum 의 첫 번째 principal axis 좌표계 projection)
    6 : pca_2
    7 : pca_3

용법.
    python scripts/viz_d008_input_channels.py \\
        --cases PENGWIN_001 PENGWIN_050 PENGWIN_100 \\
        --out_dir viz/exp4_spatial_moments_channels
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


CHANNEL_NAMES = [
    ("CT", "gray", -1000, 1500),
    ("P_sacrum", "viridis", 0.0, 1.0),
    ("rel_z", "RdBu_r", -0.5, 0.5),
    ("rel_y", "RdBu_r", -0.5, 0.5),
    ("rel_x", "RdBu_r", -0.5, 0.5),
    ("pca_1", "RdBu_r", -3.0, 3.0),
    ("pca_2", "RdBu_r", -3.0, 3.0),
    ("pca_3", "RdBu_r", -3.0, 3.0),
]


def pick_slice_with_sacrum(p_sacrum: np.ndarray) -> int:
    """sacrum 신호가 가장 강한 axial slice 의 z-index."""
    sums = p_sacrum.sum(axis=(1, 2))
    if sums.sum() == 0:
        return p_sacrum.shape[0] // 2
    return int(np.argmax(sums))


def render_case(case_id: str, d008_dir: Path, label_dir: Path, out_dir: Path) -> None:
    channels = []
    for c in range(8):
        path = d008_dir / "imagesTr" / f"{case_id}_{c:04d}.nii.gz"
        if not path.exists():
            print(f"  [skip] {case_id}: channel {c} 없음 ({path})")
            return
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        channels.append(arr)

    label_path = label_dir / f"{case_id}.nii.gz"
    label_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))) if label_path.exists() else None

    p_sacrum = channels[1]
    z = pick_slice_with_sacrum(p_sacrum)
    print(f"  [case] {case_id} — axial slice z={z}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f"PENGWIN {case_id} — Dataset008 Spatial Moments inputs (axial slice z={z})",
        fontsize=14, fontweight="bold",
    )

    for c, (name, cmap, vmin, vmax) in enumerate(CHANNEL_NAMES):
        row, col = divmod(c, 5)
        ax = axes[row, col]
        ax.imshow(channels[c][z], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"ch {c}: {name}", fontsize=11)
        ax.axis("off")

    # 마지막 두 칸: label overlay + CT, label 자체
    ct_slice = channels[0][z]
    ax = axes[1, 3]
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=1500)
    if label_arr is not None:
        ax.imshow(
            np.ma.masked_equal(label_arr[z], 0),
            cmap=plt.get_cmap("tab10"),
            vmin=0, vmax=4,
            alpha=0.5,
        )
    ax.set_title("CT + 4-class label overlay", fontsize=11)
    ax.axis("off")

    ax = axes[1, 4]
    if label_arr is not None:
        ax.imshow(
            np.ma.masked_equal(label_arr[z], 0),
            cmap=plt.get_cmap("tab10"),
            vmin=0, vmax=4,
        )
    ax.set_title("Label (bg=0, sac=1, LH=2, RH=3)", fontsize=11)
    ax.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_channels.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["PENGWIN_001", "PENGWIN_050", "PENGWIN_100", "PENGWIN_200"])
    parser.add_argument("--out_dir", type=Path, default=Path("viz/exp4_spatial_moments_channels"))
    args = parser.parse_args()

    raw = Path(env("nnUNet_raw"))
    d008 = raw / "Dataset008_PENGWIN_SpatialMoments"
    label_dir = d008 / "labelsTr"

    print(f"[setup] D008 = {d008}")
    print(f"[setup] cases = {args.cases}")
    print(f"[setup] out = {args.out_dir}")

    for case_id in args.cases:
        render_case(case_id, d008, label_dir, args.out_dir)

    print(f"\nDONE — wrote {len(args.cases)} multi-panel PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
