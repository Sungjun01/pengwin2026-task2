"""PER-CASE SEAM DIAGNOSTIC (GPU-free, argmax): at the seams where v18 MERGES two GT
fragments, what did v18's border-core argmax predict — BORDER (correct, but decode
failed to use it) or CORE (the seam filled in as solid = prediction failure)?

Uses Aseg fold_0 held-out argmax predictions (the honest failures) + GT instances.
Answers the first-order split:
  - merged-seam predicted mostly CORE  -> model under-predicts border at seam (PREDICTION problem)
  - merged-seam predicted some BORDER  -> border present in argmax but decode merged anyway
                                          (DECODE problem: band too thin/discontinuous for 26-CC)
(Soft-prob refinement "border prob low vs present-but-below-threshold" needs probabilities -> GPU.)
"""
import sys, os, glob, argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from skimage.segmentation import watershed

REPO = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
sys.path.insert(0, REPO)
from scripts.border_core_conversion import CORE_CC_STRUCTURE

GT_DIR = f"{REPO}/predictions/gt_instance"
ASEG_VAL = glob.glob("/home/schoaq/taehee/nnUNet_data/results/Dataset016*/nnUNetTrainerBorderWeightedAseg_w40__*/fold_0/validation")[0]
# bone -> (gt lo, hi, core_class, border_class)
BONES = [("sacrum", 1, 50, 1, 2), ("leftHip", 51, 100, 3, 4), ("rightHip", 101, 150, 5, 6)]
STRUCT = CORE_CC_STRUCTURE
MIN_CC_MM3 = 1000.0


def decode(sem, spz, vv):
    out = np.zeros(sem.shape, np.int32); nxt = 1
    for bn, lo, hi, core_c, bord_c in BONES:
        bone_full = (sem == core_c) | (sem == bord_c)
        if not bone_full.any():
            continue
        idx = np.where(bone_full)
        sl = tuple(slice(idx[d].min(), idx[d].max()+1) for d in range(3))
        core = (sem[sl] == core_c); bone = bone_full[sl]
        markers, n = ndi.label(core, structure=STRUCT)
        if n == 0:
            continue
        dist = ndi.distance_transform_edt(bone, sampling=spz)
        ws = watershed(-dist, markers=markers, mask=bone)
        osl = out[sl]
        for m in range(1, n+1):
            comp = ws == m
            if comp.sum()*vv < MIN_CC_MM3:
                continue
            osl[comp] = nxt; nxt += 1
    return out


def remove_small(lbl, vv):
    vals, cnts = np.unique(lbl, return_counts=True)
    drop = vals[(vals > 0) & (cnts*vv < MIN_CC_MM3)]
    if drop.size:
        keep = np.ones(int(lbl.max())+1, bool); keep[drop] = False
        lbl = lbl*keep[lbl]
    return lbl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=34); a = ap.parse_args()
    sems = sorted(glob.glob(f"{ASEG_VAL}/*.nii.gz"))[:a.n]
    tot = {"border": 0, "core": 0, "bg": 0}
    n_merged_seams = 0; n_cases_with_merge = 0; per_case = []
    for sf in sems:
        cid = os.path.basename(sf).split(".")[0]
        gf = f"{GT_DIR}/{cid}.nii.gz"
        if not os.path.exists(gf):
            continue
        sim = sitk.DICOMOrient(sitk.ReadImage(sf), "LPS")
        gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
        sem = sitk.GetArrayFromImage(sim).astype(np.int32)
        gt = sitk.GetArrayFromImage(gim).astype(np.int32)
        if sem.shape != gt.shape:
            continue
        sx, sy, sz = gim.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        pred = decode(sem, spz, vv)
        gt = remove_small(gt, vv)
        cb = {"border": 0, "core": 0, "bg": 0}; cmerge = 0
        for bn, lo, hi, core_c, bord_c in BONES:
            # GT frags in this bone
            gids = [int(v) for v in np.unique(gt) if lo <= v <= hi]
            if len(gids) < 2:
                continue
            # which pred instance does each GT frag map to (max overlap)?
            gmap = {}
            for g in gids:
                gm = gt == g
                pv = pred[gm]; pv = pv[pv > 0]
                gmap[g] = int(np.bincount(pv).argmax()) if pv.size else 0
            # merged groups: GT frags sharing the same nonzero pred id
            from collections import defaultdict
            groups = defaultdict(list)
            for g, p in gmap.items():
                if p > 0:
                    groups[p].append(g)
            for p, gg in groups.items():
                if len(gg) < 2:
                    continue
                # for each merged pair, the GT seam band = frag_i dilated ∩ frag_j
                for ii in range(len(gg)):
                    for jj in range(ii+1, len(gg)):
                        a_ = gt == gg[ii]; b_ = gt == gg[jj]
                        band = ndi.binary_dilation(a_, STRUCT, 1) & ndi.binary_dilation(b_, STRUCT, 1) & ((gt >= lo) & (gt <= hi))
                        if band.sum() < 5:
                            continue
                        cmerge += 1
                        sv = sem[band]
                        cb["border"] += int(np.sum(sv == bord_c))
                        cb["core"] += int(np.sum(sv == core_c))
                        cb["bg"] += int(np.sum((sv != bord_c) & (sv != core_c)))
        for k in tot:
            tot[k] += cb[k]
        if cmerge:
            n_cases_with_merge += 1; n_merged_seams += cmerge
            s = cb["border"] + cb["core"] + cb["bg"]
            per_case.append((cid, cmerge, cb["border"]/max(s, 1), cb["core"]/max(s, 1)))
    print(f"=== merged-seam border-prediction diagnostic (Aseg fold_0 held-out, argmax) ===", flush=True)
    print(f"cases with >=1 merged seam: {n_cases_with_merge}, total merged seams: {n_merged_seams}")
    s = sum(tot.values())
    if s:
        print(f"\nAt the GT seams where v18 MERGED fragments, v18 argmax predicted:")
        print(f"  BORDER (correct, seam marked): {tot['border']/s:.1%}  ({tot['border']} vox)")
        print(f"  CORE   (seam filled solid):    {tot['core']/s:.1%}  ({tot['core']} vox)")
        print(f"  bg/other:                      {tot['bg']/s:.1%}  ({tot['bg']} vox)")
        print(f"\nINTERPRETATION:")
        bf = tot['border']/s
        if bf < 0.20:
            print(f"  -> seam predicted mostly CORE ({tot['core']/s:.0%}) = model UNDER-PREDICTS border at seam")
            print(f"     => PREDICTION problem (model fills seam as solid). Lever: better border prediction.")
        else:
            print(f"  -> border IS predicted at {bf:.0%} of merged seam, yet cores still merged")
            print(f"     => DECODE problem (border band too thin/discontinuous for 26-CC). Lever: soft-prob/consensus decode.")
        print(f"     (Refinement of 'low prob vs present-but-below-threshold' needs softmax -> GPU re-predict.)")
    print("\nworst cases (cid, #seams, border%, core%):")
    for cid, m, bfr, cfr in sorted(per_case, key=lambda x: x[2])[:10]:
        print(f"  {cid}: {m} seams  border={bfr:.0%} core={cfr:.0%}")
    print("SEAM_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
