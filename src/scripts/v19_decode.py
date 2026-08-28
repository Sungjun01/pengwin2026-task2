"""V19 — separation-focused, region-aware border-core decode (GPU-free).

Diagnosis (report_102): on cases v18 merges, the model ALREADY predicts the
separating borders, but the border band is
discontinuous (42% core gaps). The canonical decode labels cores with
26-connectivity, so it bridges two distinct cores through those 1–2 voxel
diagonal gaps -> under-segmentation (recall 0.762, merge 2.09/case).

V19 fixes this at decode time only (no retraining):
  1. Connectivity reduction (6-conn) when labeling core seeds -> no diagonal
     bridge through a 1-voxel gap.
  2. Optional mild core erosion -> breaks thin (1–2 voxel) face bridges, with a
     thin-fragment guard so we neve on 53.8% of the seam voxelr erode a small fragment out of existence.
  3. Region-aware per-bone parameters (report_101/102): the sacrum is low-density
     with weak seams -> separate more aggressively; the dense hip (acetabulum/
     ilium, >300 HU solid bone) -> be conservative so we don't over-split a real
     solid mass.

After seeding, a distance-transform watershed fills the eroded shell back over
the full bone mask (core+border), so erosion costs no surface. Sub-threshold
fragments (<min_cc_mm3) are removed to match the challenge scoring (≥500–1000mm³).

The decode is faithful to the canonical pipeline: ``baseline_config()`` (26-conn,
no erosion) reproduces v18's decode, so V19 vs v18 is an apples-to-apples
decode-only comparison on the same predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed

# 7-class pelvic border-core encoding: (bone_name, label_lo, label_hi, core_class, border_class)
BONES = [
    ("sacrum", 1, 50, 1, 2),
    ("leftHip", 51, 100, 3, 4),
    ("rightHip", 101, 150, 5, 6),
]

_S6 = ndi.generate_binary_structure(3, 1)
_S26 = ndi.generate_binary_structure(3, 3)

DEFAULT_MIN_CC_MM3 = 1000.0


@dataclass
class BoneParams:
    """Per-bone decode parameters."""
    connectivity: int = 6           # 6 (V19 separation) or 26 (canonical)
    erode_iters: int = 0            # core erosion before CC labeling
    min_cc_mm3: float = DEFAULT_MIN_CC_MM3
    # thin-fragment guard: do NOT erode a core whose max interior half-thickness
    # (in voxels) is <= this, so rami / thin sheets survive erosion.
    protect_thickness_vox: float = 2.0
    # border-aware separation (report_102): close the predicted border band by
    # this many dilations to bridge its 42% gaps, then use it as a WALL that cuts
    # core seeds ONLY where border is predicted. 0 disables (blind connectivity).
    border_close_iters: int = 0


@dataclass
class V19Config:
    default: BoneParams = field(default_factory=BoneParams)
    bones: Dict[str, BoneParams] = field(default_factory=dict)

    def params_for_bone(self, bone_name: str) -> BoneParams:
        return self.bones.get(bone_name, self.default)


def baseline_config(min_cc_mm3: float = DEFAULT_MIN_CC_MM3) -> V19Config:
    """Reproduces v18's canonical decode: 26-connectivity, no erosion."""
    p = BoneParams(connectivity=26, erode_iters=0, min_cc_mm3=min_cc_mm3,
                   protect_thickness_vox=0.0)
    return V19Config(default=p, bones={})


def v19_config(min_cc_mm3: float = DEFAULT_MIN_CC_MM3) -> V19Config:
    """V19 region-aware separation decode.

    - sacrum: aggressive separation (6-conn + erode 1) — weak/low-density seams.
    - dense hip (L/R): conservative (6-conn, no erosion) — don't over-split solid
      acetabulum/ilium; connectivity fix alone removes diagonal bridges.
    """
    sacrum = BoneParams(connectivity=6, erode_iters=1, min_cc_mm3=min_cc_mm3,
                        protect_thickness_vox=2.0)
    hip = BoneParams(connectivity=6, erode_iters=0, min_cc_mm3=min_cc_mm3,
                     protect_thickness_vox=2.0)
    return V19Config(default=hip, bones={"sacrum": sacrum, "leftHip": hip, "rightHip": hip})


def v19_border_aware_config(min_cc_mm3: float = DEFAULT_MIN_CC_MM3,
                            border_close_iters: int = 1) -> V19Config:
    """V19 border-aware separation: cut cores ONLY where the model predicts a
    border, closing the band's 42% gaps by ``border_close_iters`` dilations.
    Uses 26-connectivity (canonical) so we don't add blind diagonal cuts — the
    border wall, not connectivity, decides the cut. This directly implements the
    report_102 prescription ("use the already-predicted 54% border").
    """
    p = BoneParams(connectivity=26, erode_iters=0, min_cc_mm3=min_cc_mm3,
                   protect_thickness_vox=2.0, border_close_iters=border_close_iters)
    return V19Config(default=p, bones={})


def _struct(connectivity: int):
    return _S6 if connectivity == 6 else _S26


def _eroded_seed(core: np.ndarray, params: BoneParams) -> np.ndarray:
    """Erode core for separation, but protect thin fragments from vanishing.

    Strategy: label the un-eroded core under the bone's connectivity. For each
    connected component, erode it; if erosion empties it OR the component is a
    thin fragment (max interior half-thickness <= protect_thickness_vox), keep
    the original (un-eroded) component instead. This guarantees we never lose a
    fragment's seed (recall guard) while still breaking thick bridges.
    """
    if params.erode_iters <= 0:
        return core
    struct = _struct(params.connectivity)
    lbl, n = ndi.label(core, structure=struct)
    if n == 0:
        return core
    out = np.zeros_like(core)
    for cc in range(1, n + 1):
        comp = lbl == cc
        # thin-fragment guard
        half_thick = float(ndi.distance_transform_edt(comp).max())
        if half_thick <= params.protect_thickness_vox:
            out |= comp
            continue
        e = ndi.binary_erosion(comp, _S6, params.erode_iters)
        out |= e if e.any() else comp
    return out


def _decode_bone(sem: np.ndarray, core_class: int, border_class: int,
                 spacing_zyx: Tuple[float, float, float], params: BoneParams):
    """Return (labeled_instances_in_bbox, bbox_slices, n_markers)."""
    bone_full = (sem == core_class) | (sem == border_class)
    if not bone_full.any():
        return None
    idx = np.where(bone_full)
    sl = tuple(slice(int(idx[d].min()), int(idx[d].max()) + 1) for d in range(3))
    core = sem[sl] == core_class
    bone = bone_full[sl]
    if not core.any():
        return None
    seed = _eroded_seed(core, params)
    # border-aware cut: subtract a gap-closed border wall from the seed so cores
    # are split ONLY where the model predicts a (possibly discontinuous) border.
    if params.border_close_iters > 0:
        border = sem[sl] == border_class
        if border.any():
            wall = ndi.binary_dilation(border, _S6, params.border_close_iters)
            cut = seed & ~wall
            # guard: never let the wall delete an entire core seed (recall guard).
            if cut.any():
                seed = cut
    markers, n = ndi.label(seed, structure=_struct(params.connectivity))
    if n == 0:
        return None
    dist = ndi.distance_transform_edt(bone, sampling=spacing_zyx)
    ws = watershed(-dist, markers=markers, mask=bone)
    return ws, sl, n


def decode_anatomy_ids(semantic: np.ndarray, core_label: int, border_label: int,
                       target_range: Tuple[int, int], voxel_volume_mm3: float,
                       spacing_zyx: Tuple[float, float, float],
                       min_volume_mm3: float = 500.0,
                       params: Optional[BoneParams] = None) -> np.ndarray:
    """V19 separation decode for ONE anatomy, producing PENGWIN instance IDs.

    Drop-in alternative to ``pivot_form_postproc.border_core_to_instances``: same
    output contract (uint16 map with IDs in [lo, hi], largest fragments first,
    sub-min_volume dropped), but uses the V19 border-aware separation so the
    26-connectivity bridge through the discontinuous border band is broken.
    """
    if params is None:
        params = BoneParams(connectivity=26, erode_iters=0,
                            min_cc_mm3=min_volume_mm3, border_close_iters=1)
    out = np.zeros_like(semantic, dtype=np.uint16)
    core = semantic == core_label
    border = semantic == border_label
    if not core.any():
        return out
    bone = core | border

    seed = _eroded_seed(core, params)
    if params.border_close_iters > 0 and border.any():
        wall = ndi.binary_dilation(border, _S6, params.border_close_iters)
        cut = seed & ~wall
        if cut.any():
            seed = cut
    markers, n = ndi.label(seed, structure=_struct(params.connectivity))
    if n == 0:
        return out
    dist = ndi.distance_transform_edt(bone, sampling=spacing_zyx)
    ws = watershed(-dist, markers=markers, mask=bone)

    # rank fragments by volume (desc), drop sub-threshold, assign lo+slot.
    sizes = []
    for m in range(1, n + 1):
        vol = int((ws == m).sum()) * voxel_volume_mm3
        if vol >= min_volume_mm3:
            sizes.append((vol, m))
    sizes.sort(reverse=True)
    lo, hi = target_range
    max_slots = hi - lo + 1
    for slot, (_vol, m) in enumerate(sizes[:max_slots]):
        out[ws == m] = lo + slot
    return out


def decode_volume(sem: np.ndarray, spacing_xyz: Tuple[float, float, float],
                  cfg: Optional[V19Config] = None) -> np.ndarray:
    """Decode a 7-class pelvic border-core SEMANTIC map into instance labels.

    sem: int array (z,y,x) with classes 0..6.
    spacing_xyz: SimpleITK-style (sx, sy, sz). Internally used as (sz, sy, sx).
    """
    if cfg is None:
        cfg = v19_config()
    sx, sy, sz = spacing_xyz
    spacing_zyx = (sz, sy, sx)
    voxel_vol = float(sx * sy * sz)
    out = np.zeros(sem.shape, np.int32)
    nxt = 1
    for bone_name, _lo, _hi, core_c, bord_c in BONES:
        params = cfg.params_for_bone(bone_name)
        res = _decode_bone(sem, core_c, bord_c, spacing_zyx, params)
        if res is None:
            continue
        ws, sl, n = res
        osl = out[sl]
        for m in range(1, n + 1):
            comp = ws == m
            if comp.sum() * voxel_vol < params.min_cc_mm3:
                continue
            osl[comp] = nxt
            nxt += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# IO — apply V19 decode to saved semantic predictions
# ─────────────────────────────────────────────────────────────────────────────
def decode_case(sem_path: Path, out_path: Path, cfg: Optional[V19Config] = None,
                reorient_lps: bool = True) -> Path:
    import SimpleITK as sitk
    img = sitk.ReadImage(str(sem_path))
    if reorient_lps:
        img = sitk.DICOMOrient(img, "LPS")
    sem = sitk.GetArrayFromImage(img).astype(np.int32)
    spacing_xyz = img.GetSpacing()
    inst = decode_volume(sem, spacing_xyz, cfg)
    out_img = sitk.GetImageFromArray(inst.astype(np.uint16))
    out_img.CopyInformation(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    return out_path


def decode_dir(sem_dir: Path, out_dir: Path, cfg: Optional[V19Config] = None,
               pattern: str = "*.nii.gz", out_suffix: str = ".mha",
               reorient_lps: bool = True) -> int:
    sem_dir, out_dir = Path(sem_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(sem_dir.glob(pattern))
    for i, f in enumerate(files, 1):
        cid = f.name.split(".")[0]
        out_path = out_dir / f"PENGWIN_{cid.replace('PENGWIN_', '')}{out_suffix}"
        decode_case(f, out_path, cfg, reorient_lps=reorient_lps)
        print(f"[{i}/{len(files)}] {cid} -> {out_path.name}", flush=True)
    return len(files)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="V19 separation-focused border-core decode")
    ap.add_argument("--sem_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", choices=["v19", "baseline"], default="v19")
    ap.add_argument("--min_cc_mm3", type=float, default=DEFAULT_MIN_CC_MM3)
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()
    cfg = baseline_config(args.min_cc_mm3) if args.config == "baseline" else v19_config(args.min_cc_mm3)
    n = decode_dir(Path(args.sem_dir), Path(args.out_dir), cfg, pattern=args.pattern)
    print(f"V19_DECODE_DONE {n} cases ({args.config})", flush=True)
