"""Build Form-Learn edge supervision from 7-class BC + GT instance (§4A.8.2b, §4A.8.6).

Nodes = prediction (or GT) core CCs per anatomy after min_volume filter.
Edges = border-adjacent core pairs. Labels from GT instance overlap assignment.

Usage:
    python3 scripts/build_form_learn_edge_dataset.py \\
        --pred_dir predictions/heldout_d015_ep999_foldall --cases 003 011

    python3 scripts/build_form_learn_edge_dataset.py \\
        --semantic_dir /path/to/labelsTr --split train \\
        --out progress/form_learn_edges_train.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.form_learn_edge import build_case_edge_rows
from scripts.measure_border_core_recovery import ANATOMY

GT_INSTANCE = _ROOT / "predictions" / "gt_instance"
DEFAULT_LABELS = Path(
    "/home/schoaq/taehee/nnUNet_data/raw/Dataset015_PENGWIN_BorderCore_OrientFixed/labelsTr"
)
HELDOUT = ("003", "011", "017", "018", "025", "034")
STRUCT26 = generate_binary_structure(3, 3)
OVERLAP_FRAC = 0.8


def _cc_kept(
    core_mask: np.ndarray, voxel_mm3: float, min_mm3: float
) -> tuple[np.ndarray, list[int]]:
    if not core_mask.any():
        return np.zeros_like(core_mask, dtype=np.int32), []
    labels, n = cc_label(core_mask, structure=STRUCT26)
    min_vox = int(np.ceil(min_mm3 / voxel_mm3))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]
    kept = [int(i + 1) for i, s in enumerate(sizes) if s >= min_vox]
    return labels, kept


def _contact_edges(
    labels: np.ndarray, kept: list[int], border_mask: np.ndarray, dilation_iters: int = 2
) -> list[tuple[int, int]]:
    if len(kept) < 2:
        return []
    border_near = binary_dilation(border_mask, structure=STRUCT26, iterations=dilation_iters)
    pairs: list[tuple[int, int]] = []
    for i, ci in enumerate(kept):
        mi = labels == ci
        for cj in kept[i + 1 :]:
            mj = labels == cj
            if np.any(border_near & mi) and np.any(border_near & mj):
                pairs.append((ci, cj))
    return pairs


def _assign_core_to_gt(
    core_mask: np.ndarray,
    ginst: np.ndarray,
    lo: int,
    hi: int,
) -> tuple[int | None, bool]:
    """Return (gt_instance_id, abstain)."""
    n_core = int(core_mask.sum())
    if n_core == 0:
        return None, True
    best_fid, best_ov = None, 0
    for fid in range(lo, hi + 1):
        ov = int((core_mask & (ginst == fid)).sum())
        if ov > best_ov:
            best_ov, best_fid = ov, fid
    if best_fid is None or best_ov == 0:
        return None, True
    if best_ov < OVERLAP_FRAC * n_core:
        return None, True
    return best_fid, False


def _edge_geom(
    labels: np.ndarray,
    ci: int,
    cj: int,
    border_mask: np.ndarray,
    voxel_mm3: float,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    mi, mj = labels == ci, labels == cj
    vi, vj = int(mi.sum()), int(mj.sum())
    contact = binary_dilation(mi | mj, structure=STRUCT26, iterations=1) & border_mask
    n_contact = int(contact.sum())
    zi, yi, xi = np.where(mi)
    zj, yj, xj = np.where(mj)
    ci_mm = np.array(
        [zi.mean() * spacing_zyx[0], yi.mean() * spacing_zyx[1], xi.mean() * spacing_zyx[2]]
    )
    cj_mm = np.array(
        [zj.mean() * spacing_zyx[0], yj.mean() * spacing_zyx[1], xj.mean() * spacing_zyx[2]]
    )
    dist_mm = float(np.linalg.norm(ci_mm - cj_mm))
    return {
        "vol_i_mm3": vi * voxel_mm3,
        "vol_j_mm3": vj * voxel_mm3,
        "vol_ratio": vi / max(vj, 1),
        "contact_voxels": n_contact,
        "contact_mm2": n_contact * (voxel_mm3 ** (2 / 3)),
        "centroid_dist_mm": dist_mm,
        "border_frac_ij": n_contact / max(n_contact + vi + vj, 1),
    }


def build_case_edges(
    case_id: str,
    sem: np.ndarray,
    ginst: np.ndarray,
    voxel_mm3: float,
    spacing_zyx: tuple[float, float, float],
    min_volume_mm3: float,
) -> list[dict]:
    rows: list[dict] = []
    cid = f"PENGWIN_{case_id}"
    for anatomy, (core_cls, border_cls, (lo, hi)) in ANATOMY.items():
        core = sem == core_cls
        border = sem == border_cls
        labels, kept = _cc_kept(core, voxel_mm3, min_volume_mm3)
        if len(kept) < 2:
            continue

        gt_map: dict[int, tuple[int | None, bool]] = {}
        for c in kept:
            gt_map[c] = _assign_core_to_gt(labels == c, ginst, lo, hi)

        for ci, cj in _contact_edges(labels, kept, border):
            gi, abst_i = gt_map[ci]
            gj, abst_j = gt_map[cj]
            if abst_i or abst_j or gi is None or gj is None:
                y_merge = None
                abstain = True
            elif gi == gj:
                y_merge = 1
                abstain = False
            else:
                y_merge = 0
                abstain = False

            geom = _edge_geom(labels, ci, cj, border, voxel_mm3, spacing_zyx)
            rows.append(
                {
                    "case_id": cid,
                    "anatomy": anatomy,
                    "core_i": ci,
                    "core_j": cj,
                    "y_merge": y_merge,
                    "abstain_flag": abstain,
                    "gt_id_i": gi,
                    "gt_id_j": gj,
                    "n_cores": len(kept),
                    **{f"e_{k}": v for k, v in geom.items()},
                }
            )
    return rows


def list_cases(split: str, semantic_dir: Path, cases: list[str] | None) -> list[str]:
    if cases:
        return cases
    if split == "heldout":
        return list(HELDOUT)
    ids = []
    for p in sorted(semantic_dir.glob("PENGWIN_*.nii.gz")):
        ids.append(p.name.removeprefix("PENGWIN_").removesuffix(".nii.gz"))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Form-Learn edge dataset (§4A.8.2b)")
    ap.add_argument("--pred_dir", type=Path, default=None, help="7-class preds (frozen ep999)")
    ap.add_argument(
        "--semantic_dir",
        type=Path,
        default=None,
        help="7-class semantic (default: pred_dir or labelsTr)",
    )
    ap.add_argument("--gt_dir", type=Path, default=GT_INSTANCE)
    ap.add_argument("--split", choices=("train", "heldout", "custom"), default="heldout")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument(
        "--probs_dir",
        type=Path,
        default=None,
        help="optional dir with PENGWIN_XXX.npz (C,Z,Y,X probabilities)",
    )
    ap.add_argument("--min_volume_mm3", type=float, default=500.0)
    ap.add_argument("--out", type=Path, default=_ROOT / "progress" / "form_learn_edges.parquet")
    args = ap.parse_args()

    sem_dir = args.semantic_dir or args.pred_dir or DEFAULT_LABELS
    cases = list_cases(args.split, sem_dir, args.cases)

    all_rows: list[dict] = []
    for case_id in cases:
        sem_path = sem_dir / f"PENGWIN_{case_id}.nii.gz"
        gt_path = args.gt_dir / f"PENGWIN_{case_id}.nii.gz"
        if not sem_path.exists() or not gt_path.exists():
            print(f"[skip] missing {case_id}", flush=True)
            continue
        img = sitk.ReadImage(str(sem_path))
        sem = sitk.GetArrayFromImage(img).astype(np.int16)
        ginst = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        sp = img.GetSpacing()
        voxel_mm3 = float(sp[0] * sp[1] * sp[2])
        spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        probs = None
        if args.probs_dir:
            npz = args.probs_dir / f"PENGWIN_{case_id}.npz"
            if npz.exists():
                import numpy as _np

                with _np.load(npz) as data:
                    probs = data["probabilities"].astype(_np.float32)
        rows = build_case_edge_rows(
            case_id, sem, ginst, voxel_mm3, spacing_zyx, args.min_volume_mm3, probs=probs
        )
        all_rows.extend(rows)
        n_merge = sum(1 for r in rows if r["y_merge"] == 1 and not r["abstain_flag"])
        print(f"[edge] {case_id}: {len(rows)} edges, merge={n_merge}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.DataFrame(all_rows)
        df.to_parquet(args.out, index=False)
    except ImportError:
        import csv

        out_csv = args.out.with_suffix(".csv")
        if all_rows:
            with out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        args.out = out_csv

    n_labeled = sum(1 for r in all_rows if r["y_merge"] is not None and not r["abstain_flag"])
    n_merge = sum(1 for r in all_rows if r["y_merge"] == 1)
    print(f"[done] {len(all_rows)} edges, labeled={n_labeled}, merge_pos={n_merge}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
