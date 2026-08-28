"""DECODE-LEVER test (GPU-free): can a LESS-AGGRESSIVE decode separate the cores that
the current 26-conn core-CC merges, using the 54%-present border that v18 ALREADY predicts?

On Aseg fold_0 held-out argmax predictions, try decode variants and score instance
recall/merge/split vs GT. Baseline (current) = core CC 26-conn + watershed.
Variants attack the '42% core-gap bridges through the border band' failure:
  conn6   : 6-connectivity core CC (no diagonal bridging through 1-voxel gaps)
  erode1  : erode core by 1 before CC (break thin bridges), then watershed back
  erode1c6: erode 1 + 6-conn (both)
Lever confirmed if a variant raises recall & lowers merge WITHOUT exploding split.
"""
import sys, os, glob, argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from skimage.segmentation import watershed

REPO = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
sys.path.insert(0, REPO)
GT_DIR = f"{REPO}/predictions/gt_instance"
ASEG_VAL = glob.glob("/home/schoaq/taehee/nnUNet_data/results/Dataset016*/nnUNetTrainerBorderWeightedAseg_w40__*/fold_0/validation")[0]
BONES = [("sacrum", 1, 50, 1, 2), ("leftHip", 51, 100, 3, 4), ("rightHip", 101, 150, 5, 6)]
S26 = ndi.generate_binary_structure(3, 3)
S6 = ndi.generate_binary_structure(3, 1)
MIN_CC_MM3 = 1000.0


def decode(sem, spz, vv, variant):
    out = np.zeros(sem.shape, np.int32); nxt = 1
    for bn, lo, hi, core_c, bord_c in BONES:
        bone_full = (sem == core_c) | (sem == bord_c)
        if not bone_full.any():
            continue
        idx = np.where(bone_full)
        sl = tuple(slice(idx[d].min(), idx[d].max()+1) for d in range(3))
        core = (sem[sl] == core_c); bone = bone_full[sl]
        cc_struct = S6 if "c6" in variant or variant == "conn6" else S26
        seed = core
        if "erode1" in variant:
            seed = ndi.binary_erosion(core, S6, 1)
            if not seed.any():
                seed = core
        markers, n = ndi.label(seed, structure=cc_struct)
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


def score(gt, pred, vv):
    gt = remove_small(gt.astype(np.int32), vv); pred = remove_small(pred.astype(np.int32), vv)
    gids = np.array(sorted(int(x) for x in np.unique(gt) if x > 0))
    pids = np.array(sorted(int(x) for x in np.unique(pred) if x > 0))
    G, P = len(gids), len(pids)
    if G == 0: return 1.0, 1.0, 1.0, 0, 0
    if P == 0: return 0.0, 0.0, 0.0, 0, 0
    gl = np.zeros(int(gt.max())+1, np.int64); gl[gids] = np.arange(1, G+1)
    pl = np.zeros(int(pred.max())+1, np.int64); pl[pids] = np.arange(1, P+1)
    cont = np.bincount((gl[gt]*(P+1)+pl[pred]).ravel(), minlength=(G+1)*(P+1)).reshape(G+1, P+1).astype(np.float64)
    inter = cont[1:, 1:]; gsz = cont[1:, :].sum(1); psz = cont[:, 1:].sum(0)
    iou = inter/np.maximum(gsz[:, None]+psz[None, :]-inter, 1.0)
    ug, up, tp = set(), set(), 0
    for v, i, j in sorted(((iou[i, j], i, j) for i in range(G) for j in range(P) if iou[i, j] >= 0.5), reverse=True):
        if i in ug or j in up: continue
        ug.add(i); up.add(j); tp += 1
    rec = tp/G; prec = tp/P; f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    sub = inter > 0.2*np.minimum(gsz[:, None], psz[None, :])
    merge = sum(max(0, sub[:, j].sum()-1) for j in range(P))
    split = sum(max(0, sub[i, :].sum()-1) for i in range(G))
    return f1, rec, prec, merge, split


def main():
    variants = ["baseline", "conn6", "erode1", "erode1c6"]
    sems = sorted(glob.glob(f"{ASEG_VAL}/*.nii.gz"))
    agg = {v: [] for v in variants}
    import time
    for sf in sems:
        cid = os.path.basename(sf).split(".")[0]
        gf = f"{GT_DIR}/{cid}.nii.gz"
        if not os.path.exists(gf):
            continue
        sim = sitk.DICOMOrient(sitk.ReadImage(sf), "LPS"); gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
        sem = sitk.GetArrayFromImage(sim).astype(np.int32); gt = sitk.GetArrayFromImage(gim).astype(np.int32)
        if sem.shape != gt.shape: continue
        sx, sy, sz = gim.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        for v in variants:
            pred = decode(sem, spz, vv, v)
            agg[v].append(score(gt, pred, vv))
    print(f"=== decode-variant comparison (Aseg fold_0 held-out, n={len(agg['baseline'])}) ===", flush=True)
    print(f"{'variant':>10} {'F1':>6} {'recall':>7} {'prec':>6} {'merge':>6} {'split':>6}")
    for v in variants:
        A = np.array(agg[v])
        print(f"{v:>10} {A[:,0].mean():>6.3f} {A[:,1].mean():>7.3f} {A[:,2].mean():>6.3f} {A[:,3].mean():>6.2f} {A[:,4].mean():>6.2f}")
    b = np.array(agg["baseline"])
    print(f"\n  baseline recall={b[:,1].mean():.3f} merge={b[:,3].mean():.2f} (= F-CIRL-on canonical decode)")
    for v in variants[1:]:
        A = np.array(agg[v]); dr = A[:,1].mean()-b[:,1].mean(); dm = A[:,3].mean()-b[:,3].mean(); ds = A[:,4].mean()-b[:,4].mean()
        verdict = "WIN" if dr > 0.02 and dm < 0 and ds < 0.5 else ("split-cost" if ds >= 0.5 else "~flat")
        print(f"  {v:>10}: Δrecall={dr:+.3f} Δmerge={dm:+.2f} Δsplit={ds:+.2f}  -> {verdict}")
    print("DECODE_VARIANTS_DONE", flush=True)


if __name__ == "__main__":
    main()
