"""PENGWIN GT (1-200 instance IDs) -> joint border-core label map (seam definition).

Border = the inter-fragment SEAM only (voxels within `t` of a *different*
fragment of the same anatomy), NOT a full erosion shell. Core = anatomy
foreground minus the seam.

Why seam, not full-shell erosion: pelvic fragments are thin curved sheets;
eroding the full outer shell shatters a single sheet into many disconnected
cores (core-CC ≫ true fragment count → over-segmentation). We only need to
separate *touching* fragments, so we carve only their contact band. A solitary
(unfractured) fragment then has zero border and stays fully connected. This is
the standard inter-instance boundary formulation (cf. U-Net 2015 cell borders),
matched to thin-sheet anatomy. See docs/plans/2026-05-23-border-core-implementation.md §0.

Class encoding:
  pelvic (7-class): bg / sacrum {core,border} / leftHip {core,border} / rightHip {core,border}
  femur  (3-class): bg / femur {core,border}

Connectivity: seam distance is Euclidean (distance_transform_edt). Downstream
core connected-components must use 26-connectivity (GT fragments are 26-connected;
6-conn would over-split thin cores). The ~2t-wide seam keeps 26-conn separation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# anatomy spec: (lo, hi, core_class, border_class) over PENGWIN instance IDs
PELVIC_ANATOMY = [(1, 50, 1, 2), (51, 100, 3, 4), (101, 150, 5, 6)]
FEMUR_ANATOMY = [(151, 200, 1, 2)]

CORE_CC_STRUCTURE = ndi.generate_binary_structure(3, 3)  # 26-connectivity for core CC


def convert_to_border_core(instance: np.ndarray, anatomy, t: int = 2) -> np.ndarray:
    """Convert a PENGWIN instance label volume to a joint border-core label map.

    instance: integer label volume (PENGWIN IDs 1-200, 0 = background).
    anatomy:  list of (lo, hi, core_class, border_class).
    t:        seam half-width in voxels (Euclidean). border = fragment voxels
              within `t` of a different same-anatomy fragment.
    Returns uint8 joint label map. A solitary fragment (no other fragment in its
    anatomy) gets no border — its whole extent is core.
    """
    out = np.zeros(instance.shape, dtype=np.uint8)
    for lo, hi, core_class, border_class in anatomy:
        anat_fg = (instance >= lo) & (instance <= hi)
        if not anat_fg.any():
            continue
        # Crop to the anatomy bounding box: all same-anatomy fragments live inside
        # it, so seam distances are exact while the EDT runs on a small subvolume.
        zz, yy, xx = np.where(anat_fg)
        sl = (slice(zz.min(), zz.max() + 1),
              slice(yy.min(), yy.max() + 1),
              slice(xx.min(), xx.max() + 1))
        sub = instance[sl]
        sub_fg = (sub >= lo) & (sub <= hi)
        border = np.zeros(sub.shape, dtype=bool)
        for frag_id in np.unique(sub[sub_fg]):
            this_frag = (sub == frag_id)
            other_frags = sub_fg & ~this_frag
            if not other_frags.any():
                continue  # solitary fragment in this anatomy -> no seam
            dist_to_others = ndi.distance_transform_edt(~other_frags)
            border |= this_frag & (dist_to_others <= t)
        out_sub = out[sl]
        out_sub[sub_fg & ~border] = core_class
        out_sub[border] = border_class
    return out
