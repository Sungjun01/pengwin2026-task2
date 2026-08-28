"""Inference helpers for SAFT-GD.

Two root-cause fixes live here:
  * The anchor prior is rebuilt PER sliding-window from the window CT, exactly as
    training builds it per patch (GT-free, identical frame at train and infer).
  * The decode actually consumes the boundary instance head and applies a size
    filter, instead of running naive connected-components on the anatomy argmax
    (which shattered noisy anatomy into ~100 spurious fragments).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.infer_voxel_mamba_case import _starts
from scripts.saft_coordinates import build_anchor_channels
from scripts.saft_model import SAFTOutput, SAFTSegmenter
from scripts.saft_signal_channels import build_signal_channels
from scripts.voxel_mamba_io import load_lps_volume, normalize_hu, save_label_volume_like
from scripts.voxel_mamba_losses import DEFAULT_AFFINITY_OFFSETS


def decode_saft_to_fragments(
    anatomy_zyx: np.ndarray,
    boundary_prob_zyx: np.ndarray,
    *,
    boundary_threshold: float = 0.5,
    min_voxels: int = 1,
) -> np.ndarray:
    """Decode dense anatomy + boundary probability into PENGWIN fragment ids.

    Within each anatomy bone the predicted inter-fragment boundary splits the
    bone into cores; the cores are grown back over the bone by nearest-core
    distance (a cheap watershed), and components smaller than `min_voxels` are
    dropped so single-voxel speckle no longer becomes a fragment.
    """
    from evaluation.label_schema import BONE_RANGES
    from scipy.ndimage import distance_transform_edt, generate_binary_structure, label as cc_label

    anatomy = np.asarray(anatomy_zyx, dtype=np.int32)
    boundary_prob = np.asarray(boundary_prob_zyx, dtype=np.float32)
    pred = np.zeros_like(anatomy, dtype=np.int32)
    structure = generate_binary_structure(3, 1)
    for bone, (lo, hi) in BONE_RANGES.items():
        bone_mask = anatomy == bone
        if not bone_mask.any():
            continue
        interior = bone_mask & (boundary_prob < float(boundary_threshold))
        cores, n_cores = cc_label(interior, structure=structure)
        if n_cores == 0:
            cores, n_cores = cc_label(bone_mask, structure=structure)
        if n_cores == 0:
            continue
        # Grow cores to fill the bone mask: assign every voxel to its nearest core.
        _, indices = distance_transform_edt(cores == 0, return_indices=True)
        grown = np.where(bone_mask, cores[tuple(indices)], 0)
        next_id = int(lo)
        for comp in range(1, int(n_cores) + 1):
            mask = grown == comp
            if int(mask.sum()) < int(min_voxels):
                continue
            if next_id > int(hi):
                break
            pred[mask] = next_id
            next_id += 1
    return pred


def affinity_watershed_decode(
    anatomy_zyx: np.ndarray,
    attractive: np.ndarray,
    offsets: tuple[tuple[int, int, int], ...] = DEFAULT_AFFINITY_OFFSETS,
    *,
    merge_threshold: float = 0.5,
    min_voxels: int = 1,
) -> np.ndarray:
    """Group foreground voxels into instances via local same-instance affinity.

    Two adjacent foreground voxels are merged when their predicted affinity for
    that offset exceeds `merge_threshold`. Connected components of the resulting
    graph are the instances -- this splits fragments that are spatially connected
    but separated by a low-affinity interface, which plain connected-components on
    the anatomy cannot do. The affinity head is directional (one channel per
    offset), so a fracture face is cut along its own axis while same-instance
    in-plane edges stay merged (the boundary head, being non-directional, cannot
    do this without shattering planar interfaces, so it is not used here).

    `attractive` is the sigmoid of the affinity head, shape (len(offsets), Z, Y, X),
    with channel k corresponding to `offsets[k]` (the training affinity offsets).
    Each instance is mapped into the PENGWIN id range of its majority anatomy
    bone; per-bone ids are assigned largest-fragment-first so the 50-id-per-bone
    cap keeps the biggest fragments. Components below `min_voxels` are dropped.
    """
    from evaluation.label_schema import BONE_RANGES
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    anatomy = np.asarray(anatomy_zyx, dtype=np.int32)
    attractive = np.asarray(attractive, dtype=np.float32)
    pred = np.zeros_like(anatomy, dtype=np.int32)
    foreground = anatomy > 0
    n_fg = int(foreground.sum())
    if n_fg == 0:
        return pred

    voxel_index = np.full(anatomy.shape, -1, dtype=np.int64)
    voxel_index[foreground] = np.arange(n_fg, dtype=np.int64)
    z, y, x = anatomy.shape

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for channel, (dz, dy, dx) in enumerate(offsets):
        src = (slice(0, z - dz), slice(0, y - dy), slice(0, x - dx))
        dst = (slice(dz, z), slice(dy, y), slice(dx, x))
        merge = foreground[src] & foreground[dst] & (attractive[channel][src] > float(merge_threshold))
        rows.append(voxel_index[src][merge])
        cols.append(voxel_index[dst][merge])

    edge_rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    edge_cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(edge_rows.shape[0], dtype=np.uint8), (edge_rows, edge_cols)),
        shape=(n_fg, n_fg),
    )
    n_comp, comp_of_voxel = connected_components(graph, directed=False)

    sizes = np.bincount(comp_of_voxel, minlength=n_comp)
    anatomy_fg = anatomy[foreground]
    n_classes = int(max(anatomy.max(), 4)) + 1
    bone_hist = np.zeros((n_comp, n_classes), dtype=np.int64)
    np.add.at(bone_hist, (comp_of_voxel, anatomy_fg), 1)
    bone_of_comp = bone_hist[:, 1:].argmax(axis=1) + 1  # majority foreground bone (1..4)

    fragment_of_comp = np.zeros(n_comp, dtype=np.int32)
    for bone, (lo, hi) in BONE_RANGES.items():
        selected = np.where((sizes >= int(min_voxels)) & (bone_of_comp == bone))[0]
        if selected.size == 0:
            continue
        selected = selected[np.argsort(-sizes[selected])]  # largest fragments first
        keep = min(selected.size, int(hi) - int(lo) + 1)
        fragment_of_comp[selected[:keep]] = np.arange(int(lo), int(lo) + keep, dtype=np.int32)

    pred[foreground] = fragment_of_comp[comp_of_voxel]
    return pred


def decode_saft_logits_to_fragments(
    output: SAFTOutput,
    min_voxels: int = 1,
    boundary_threshold: float = 0.5,
) -> np.ndarray:
    anatomy = torch.argmax(output.anatomy_logits[0].detach().cpu(), dim=0).numpy().astype(np.int32)
    boundary_prob = torch.sigmoid(output.boundary_logits[0, 0].detach().cpu()).numpy().astype(np.float32)
    return decode_saft_to_fragments(
        anatomy, boundary_prob, boundary_threshold=boundary_threshold, min_voxels=min_voxels
    )


def saft_predict_volume(
    model: SAFTSegmenter,
    image_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    patch_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    device: torch.device | str = "cuda",
    signal_channels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window predict, rebuilding the GT-free anchor prior per window.

    Returns averaged `(anatomy_logits[C,Z,Y,X], boundary_prob[Z,Y,X],
    affinity_prob[A,Z,Y,X])`, where affinity channels match the training offsets.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    z, y, x = (int(s) for s in image_zyx.shape)
    dz, dy, dx = (int(p) for p in patch_size)
    anatomy_sum: np.ndarray | None = None
    affinity_sum: np.ndarray | None = None
    boundary_sum = np.zeros((z, y, x), dtype=np.float32)
    weight = np.zeros((z, y, x), dtype=np.float32)
    with torch.no_grad():
        for z0 in _starts(z, dz, overlap):
            for y0 in _starts(y, dy, overlap):
                for x0 in _starts(x, dx, overlap):
                    z1, y1, x1 = min(z0 + dz, z), min(y0 + dy, y), min(x0 + dx, x)
                    ct = image_zyx[z0:z1, y0:y1, x0:x1]
                    pad = ((0, dz - (z1 - z0)), (0, dy - (y1 - y0)), (0, dx - (x1 - x0)))
                    ct_padded = np.pad(ct, pad, mode="constant", constant_values=-1.0)
                    priors = build_anchor_channels(ct_padded, spacing_zyx)
                    parts = [ct_padded[None], priors]
                    if signal_channels:
                        parts.append(build_signal_channels(ct_padded, spacing_zyx))
                    inp = np.concatenate(parts, axis=0).astype(np.float32)
                    patch = torch.from_numpy(inp)[None].to(device)
                    out = model(patch)
                    anatomy = out.anatomy_logits[0].float().cpu().numpy()
                    boundary = torch.sigmoid(out.boundary_logits[0, 0]).float().cpu().numpy()
                    affinity = torch.sigmoid(out.affinity_logits[0]).float().cpu().numpy()
                    if anatomy_sum is None:
                        anatomy_sum = np.zeros((anatomy.shape[0], z, y, x), dtype=np.float32)
                        affinity_sum = np.zeros((affinity.shape[0], z, y, x), dtype=np.float32)
                    anatomy_sum[:, z0:z1, y0:y1, x0:x1] += anatomy[:, : z1 - z0, : y1 - y0, : x1 - x0]
                    affinity_sum[:, z0:z1, y0:y1, x0:x1] += affinity[:, : z1 - z0, : y1 - y0, : x1 - x0]
                    boundary_sum[z0:z1, y0:y1, x0:x1] += boundary[: z1 - z0, : y1 - y0, : x1 - x0]
                    weight[z0:z1, y0:y1, x0:x1] += 1.0
    assert anatomy_sum is not None and affinity_sum is not None
    weight = np.clip(weight, 1.0, None)
    return anatomy_sum / weight[None], boundary_sum / weight, affinity_sum / weight[None]


def _load_saft_model_for_inference(
    checkpoint_path: str | Path,
    device_obj: torch.device,
) -> SAFTSegmenter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = SAFTSegmenter(
        in_channels=int(config.get("in_channels", 6)),
        backbone=config.get("backbone", "swin_unetr"),
        feature_channels=int(config.get("feature_channels", 24)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device_obj).eval()


def infer_saft_case_from_paths(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    patch_size: tuple[int, int, int] | None = None,
    min_voxels: int = 20,
    device: str = "auto",
    decode: str = "affinity",
    merge_threshold: float = 0.5,
) -> Path:
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = _load_saft_model_for_inference(checkpoint_path, device_obj)
    if patch_size is None:
        cfg_patch = config.get("patch_size")
        patch_size = tuple(int(v) for v in cfg_patch) if cfg_patch else (96, 96, 96)
    volume = load_lps_volume(image_path, dtype=np.float32)
    image = normalize_hu(volume.array)
    anatomy_logits, boundary_prob, affinity_prob = saft_predict_volume(
        model,
        image,
        volume.spacing_zyx,
        patch_size=patch_size,
        device=device_obj,
        signal_channels=bool(config.get("signal_channels", False)),
    )
    anatomy = np.argmax(anatomy_logits, axis=0).astype(np.int32)
    if decode == "affinity":
        pred = affinity_watershed_decode(
            anatomy, affinity_prob, merge_threshold=merge_threshold, min_voxels=min_voxels
        )
    elif decode == "boundary":
        pred = decode_saft_to_fragments(anatomy, boundary_prob, min_voxels=min_voxels)
    else:
        raise ValueError(f"unknown decode mode: {decode!r} (use 'affinity' or 'boundary')")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_label_volume_like(pred, image_path, output_path)
    return output_path


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be like 96,96,96")
    return (parts[0], parts[1], parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=_parse_patch_size, default=None)
    parser.add_argument("--min-voxels", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode", choices=("affinity", "boundary"), default="affinity")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    args = parser.parse_args()
    out = infer_saft_case_from_paths(
        args.checkpoint,
        args.image,
        args.output,
        args.patch_size,
        args.min_voxels,
        args.device,
        decode=args.decode,
        merge_threshold=args.merge_threshold,
    )
    print(json.dumps({"event": "done", "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
