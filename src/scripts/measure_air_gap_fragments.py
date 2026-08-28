"""Air-gap fragment ratio over the 340-case trainset (Risk 5 / border-adjacency limitation).

The border-adjacency post-proc (Task 5 §5.1) keeps a core CC only if it is seam-bounded
OR the largest in its anatomy. A *real* fragment separated from all its siblings by a
background air-gap (no contact -> no seam in the border-core GT) is therefore at risk of
being wrongly dropped when it is not the largest. This script quantifies how common that is.

Per fragment, in a multi-fragment anatomy, we compute the minimum VOXEL distance to the
nearest other same-anatomy fragment (matches the t=2 seam conversion, which is voxel-based):
  - contact / seam-bounded : min_dist <= t (=2)  -> safe (will have a seam)
  - air-gap isolated        : min_dist  > t       -> at risk if not the largest

Decision rule (per the plan):
  at-risk fraction < 5%  -> documented limitation, no code change
  at-risk fraction > 5%  -> add an air-gap exception to _border_adjacency_keep

용법: python3 scripts/measure_air_gap_fragments.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

LABEL_GLOB = "raw_data/PENGWIN_train/*/label.mha"
ANAT = [("sacrum", 1, 50), ("leftHip", 51, 100), ("rightHip", 101, 150), ("femur", 151, 200)]
T_SEAM = 2          # seam conversion threshold (voxels)
MIN_MM3 = 1000.0    # eval-eligible fragment (matches recovery measurement)


def bbox(mask):
    zz, yy, xx = np.where(mask)
    return (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1),
            slice(xx.min(), xx.max() + 1))


def main():
    files = sorted(glob.glob(LABEL_GLOB))
    if not files:
        raise SystemExit(f"no labels under {LABEL_GLOB}")

    min_dists_vox = []      # per at-risk-eligible fragment (multi-frag anatomy, non-largest)
    min_dists_mm = []
    n_frag = 0              # eval-eligible fragments in multi-fragment anatomies
    n_nonlargest = 0        # ... that are not the largest in their anatomy
    n_airgap = 0            # ... air-gap isolated (min_dist > t)
    n_airgap_nonlargest = 0 # ... air-gap AND non-largest = the actual regression risk
    n_cases = 0

    for f in files:
        cid = os.path.basename(os.path.dirname(f))
        img = sitk.ReadImage(f)
        arr = sitk.GetArrayFromImage(img)
        sx, sy, sz = img.GetSpacing()
        vmm = float(sx * sy * sz)
        vox_mm = float((sx + sy + sz) / 3.0)  # mean spacing for vox->mm of the min distance
        n_cases += 1

        for _, lo, hi in ANAT:
            anat_fg = (arr >= lo) & (arr <= hi)
            if not anat_fg.any():
                continue
            sl = bbox(anat_fg)
            sub = arr[sl]
            sub_fg = (sub >= lo) & (sub <= hi)
            frag_ids = [fid for fid in np.unique(sub[sub_fg])]
            # eval-eligible fragments only
            sizes = {fid: int((sub == fid).sum()) for fid in frag_ids}
            eligible = [fid for fid in frag_ids if sizes[fid] * vmm >= MIN_MM3]
            if len(eligible) < 2:
                continue  # solitary anatomy -> the one fragment is always the largest -> safe
            largest = max(eligible, key=lambda fid: sizes[fid])
            for fid in eligible:
                this_frag = (sub == fid)
                others = sub_fg & ~this_frag
                if not others.any():
                    continue
                md = float(distance_transform_edt(~others)[this_frag].min())
                n_frag += 1
                is_largest = (fid == largest)
                if not is_largest:
                    n_nonlargest += 1
                    min_dists_vox.append(md)
                    min_dists_mm.append(md * vox_mm)
                if md > T_SEAM:
                    n_airgap += 1
                    if not is_largest:
                        n_airgap_nonlargest += 1

    print(f"=== air-gap fragment measurement ({n_cases} cases, eval-eligible >= {MIN_MM3:.0f} mm3) ===")
    print(f"eval-eligible fragments in multi-fragment anatomies : {n_frag}")
    print(f"  of which NON-largest (could be dropped)            : {n_nonlargest}")
    print(f"  air-gap isolated (min dist > {T_SEAM} vox)              : {n_airgap}  ({100*n_airgap/max(n_frag,1):.1f}% of all)")
    print(f"  AIR-GAP & NON-LARGEST (= regression risk)          : {n_airgap_nonlargest}  ({100*n_airgap_nonlargest/max(n_frag,1):.2f}% of all frags)")
    if n_nonlargest:
        print(f"     (= {100*n_airgap_nonlargest/n_nonlargest:.1f}% of non-largest frags)")
    if min_dists_vox:
        d = np.array(min_dists_vox)
        mm = np.array(min_dists_mm)
        print(f"\nmin inter-fragment distance of NON-largest frags (voxels):")
        for p in (10, 25, 50, 75, 90, 95, 99):
            print(f"   p{p:>2}: {np.percentile(d, p):5.2f} vox   {np.percentile(mm, p):5.2f} mm")
        print(f"   contact (<= {T_SEAM} vox): {100*(d <= T_SEAM).mean():.1f}%   air-gap (> {T_SEAM} vox): {100*(d > T_SEAM).mean():.1f}%")
    verdict = "DOCUMENTED LIMITATION (no code change)" if (n_airgap_nonlargest / max(n_frag, 1) < 0.05) \
        else "ADD air-gap exception to _border_adjacency_keep"
    print(f"\nDECISION (threshold 5% of all eval-eligible frags): {verdict}")


if __name__ == "__main__":
    main()
