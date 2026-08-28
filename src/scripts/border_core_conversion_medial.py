"""D016 variant — medial-axis adaptive seam half-width.

Identical seam semantics to ``convert_to_border_core`` (border = inter-fragment
contact band, solitary fragments stay all-core), but the seam half-width ``t`` is
chosen per-fragment based on the fragment's medial-axis distance (half-thickness
at the thickest point). The motivation, from PENGWIN 2024 1st-place ABBC
[Sang et al. 2025 arXiv:2504.02382, MIC-DKFZ Johannsen et al.]:

  > introduced a novel dynamic boundary thickness adjustment using the medial
  > axis transform, ensuring that thin fragments remain connected within the
  > core region.

Why ``per-fragment uniform t (scaled by max interior distance)`` and not voxel-wise:
  - Per-voxel scaling using local interior_dist gives ``t small near surface,
    large in deep interior'' — but contact-band voxels are by definition near
    surface, so they would always get t≈1 regardless of overall fragment
    thickness. That's not the behaviour we want.
  - The real signal is ``how thick is this fragment as a whole'' — a 2-voxel-thick
    sheet vs a 20-voxel-thick mass need different seam widths. ``max(interior_dist)
    over the fragment'' is the cleanest scalar that captures that.

Formula:
    half_thickness = max(distance_transform_edt(this_fragment))   # voxels
    local_t = clip( ceil(half_thickness / DIVISOR), T_MIN, T_MAX )

Defaults (DIVISOR=3, T_MIN=1, T_MAX=3) give:
    half_thickness <= 1.5  → t = 1   (very thin sheet, save the core)
    half_thickness <= 4.5  → t = 2   (medium)
    half_thickness  > 4.5  → t = 3   (thick — wider seam, cleaner CC)

These are the divisor / floor / cap that the original ABBC paper does not
publish exactly; tunable via the function arguments and a small ablation grid
once D016 is preprocessed.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi

from scripts.border_core_conversion import (  # re-export for callers
    PELVIC_ANATOMY,
    FEMUR_ANATOMY,
    CORE_CC_STRUCTURE,
)


def convert_to_border_core_medial(
    instance: np.ndarray,
    anatomy,
    divisor: float = 3.0,
    t_min: int = 1,
    t_max: int = 3,
) -> np.ndarray:
    """Same as ``convert_to_border_core`` but seam half-width is per-fragment
    medial-axis adaptive (see module docstring).

    instance: integer label volume (PENGWIN IDs 1-200, 0 = background).
    anatomy:  list of (lo, hi, core_class, border_class).
    divisor / t_min / t_max: tunables for adaptive t (see module docstring).
    Returns uint8 joint label map with the same 7-class (pelvic) or 3-class (femur)
    encoding as the D015 ``convert_to_border_core``.
    """
    out = np.zeros(instance.shape, dtype=np.uint8)
    for lo, hi, core_class, border_class in anatomy:
        anat_fg = (instance >= lo) & (instance <= hi)
        if not anat_fg.any():
            continue
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
                continue  # solitary fragment — no seam, all core
            # Interior distance map for THIS fragment: distance from each voxel
            # to the fragment's own boundary. Max equals half local thickness
            # at the thickest point (≈ medial-axis radius).
            interior_dist = ndi.distance_transform_edt(this_frag)
            half_thickness = float(interior_dist.max())
            local_t = max(t_min, min(int(math.ceil(half_thickness / divisor)), t_max))
            dist_to_others = ndi.distance_transform_edt(~other_frags)
            border |= this_frag & (dist_to_others <= local_t)
        out_sub = out[sl]
        out_sub[sub_fg & ~border] = core_class
        out_sub[border] = border_class
    return out


# Convenience: keep the same call signature as D015 builder by accepting an
# optional ``t_fixed`` argument that, if given, routes to the legacy fixed-t
# function (used by tests to compare adaptive vs fixed on the same shape).
def convert_to_border_core_adaptive_or_fixed(
    instance: np.ndarray,
    anatomy,
    t_fixed: int | None = None,
    divisor: float = 3.0,
    t_min: int = 1,
    t_max: int = 3,
) -> np.ndarray:
    if t_fixed is not None:
        from scripts.border_core_conversion import convert_to_border_core
        return convert_to_border_core(instance, anatomy, t=t_fixed)
    return convert_to_border_core_medial(
        instance, anatomy, divisor=divisor, t_min=t_min, t_max=t_max
    )
