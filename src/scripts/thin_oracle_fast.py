"""THIN-BORDER ORACLE CEILING (fast, per-fragment crops). Zero training.

Does a THINNER / per-voxel-medial border-core separate MORE fragments than D016's
per-fragment-scalar border? Build border-core from GT 3 ways, run the SAME canonical
decode (core CC + watershed), score instance recall/F1 vs GT (official 1000mm3 cleanup).

Policies (border = voxels of a fragment within t of OTHER fragments):
  D016   : t = clip(ceil(edt(fragment).max()/3), 1, 3)   per-FRAGMENT scalar (current production)
  t1     : t = 1                                          uniform thin
  pervox : t = clip(ceil(maxfilter(edt(fragment),5)/3),1,3) per-VOXEL local medial radius (the fix)
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
BONES = [("sacrum", 1, 50, 1, 2), ("leftHip", 51, 100, 3, 4), ("rightHip", 101, 150, 5, 6)]
MIN_CC_MM3 = 1000.0
STRUCT = CORE_CC_STRUCTURE
MARGIN = 4


def build_bc(gt, policy):
    out = np.zeros(gt.shape, np.uint8)
    for bn, lo, hi, core_c, bord_c in BONES:
        fg = (gt >= lo) & (gt <= hi)
        if not fg.any():
            continue
        out[fg] = core_c
        ids = [int(v) for v in np.unique(gt[fg])]
        for fid in ids:
            idx = np.where(gt == fid)
            sl = tuple(slice(max(0, idx[d].min()-MARGIN), min(gt.shape[d], idx[d].max()+1+MARGIN)) for d in range(3))
            this = (gt[sl] == fid)
            others = fg[sl] & (gt[sl] != fid)
            if not others.any():
                continue
            d_others = ndi.distance_transform_edt(~others)
            if policy == "t1":
                t = 1.0
            elif policy == "D016":
                interior = ndi.distance_transform_edt(this)
                t = float(np.clip(np.ceil(interior.max()/3.0), 1, 3))
            elif policy == "pervox":
                interior = ndi.distance_transform_edt(this)
                t = np.clip(np.ceil(ndi.maximum_filter(interior, size=5)/3.0), 1, 3)
            border = this & (d_others <= t)
            out_sl = out[sl]
            out_sl[border] = bord_c
    return out


def decode(bc, spz):
    out = np.zeros(bc.shape, np.int32); nxt = 1
    for bn, lo, hi, core_c, bord_c in BONES:
        bone_full = (bc == core_c) | (bc == bord_c)
        if not bone_full.any():
            continue
        idx = np.where(bone_full)
        sl = tuple(slice(idx[d].min(), idx[d].max()+1) for d in range(3))   # bone bbox crop
        core = (bc[sl] == core_c); bone = bone_full[sl]
        markers, n = ndi.label(core, structure=STRUCT)
        if n == 0:
            continue
        dist = ndi.distance_transform_edt(bone, sampling=spz)
        ws = watershed(-dist, markers=markers, mask=bone)
        osl = out[sl]
        for m in range(1, n+1):
            osl[ws == m] = nxt; nxt += 1
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
    if G == 0: return 1.0, 1.0, 1.0
    if P == 0: return 0.0, 0.0, 0.0
    gl = np.zeros(int(gt.max())+1, np.int64); gl[gids] = np.arange(1, G+1)
    pl = np.zeros(int(pred.max())+1, np.int64); pl[pids] = np.arange(1, P+1)
    cont = np.bincount((gl[gt]*(P+1)+pl[pred]).ravel(), minlength=(G+1)*(P+1)).reshape(G+1, P+1).astype(np.float64)
    inter = cont[1:,1:]; gsz = cont[1:,:].sum(1); psz = cont[:,1:].sum(0)
    iou = inter/np.maximum(gsz[:,None]+psz[None,:]-inter, 1.0)
    ug, up, tp = set(), set(), 0
    for v,i,j in sorted(((iou[i,j],i,j) for i in range(G) for j in range(P) if iou[i,j]>=0.5), reverse=True):
        if i in ug or j in up: continue
        ug.add(i); up.add(j); tp += 1
    rec = tp/G; prec = tp/P; return (2*prec*rec/(prec+rec) if prec+rec else 0.0), rec, prec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="+", required=True); a = ap.parse_args()
    POL = ["D016", "t1", "pervox"]
    agg = {p: [] for p in POL}
    print(f"{'case':>5} nGT | " + " | ".join(f"{p:>6}" for p in POL) + "   (recall)", flush=True)
    import time
    for num in a.cases:
        gf = f"{GT_DIR}/PENGWIN_{num}.nii.gz"
        if not os.path.exists(gf): continue
        gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
        gt = sitk.GetArrayFromImage(gim).astype(np.int32); gt[(gt < 1)|(gt > 150)] = 0
        if not (gt > 0).any(): continue
        gi = np.where(gt > 0)  # crop to pelvic bbox globally (huge speedup)
        gsl = tuple(slice(max(0, gi[d].min()-MARGIN-1), gi[d].max()+MARGIN+2) for d in range(3))
        gt = gt[gsl]
        sx, sy, sz = gim.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        ng = len([v for v in np.unique(gt) if v > 0]); t0 = time.time(); row = {}
        for p in POL:
            bc = build_bc(gt, p); pred = decode(bc, spz)
            f1, rec, prec = score(gt, pred, vv); agg[p].append((f1, rec, prec)); row[p] = rec
        print(f"{num:>5} {ng:>3} | " + " | ".join(f"{row[p]:>6.3f}" for p in POL) + f"   ({time.time()-t0:.0f}s)", flush=True)
    print("\n===== ORACLE CEILING AGGREGATE (n=%d pelvic) =====" % len(agg["D016"]))
    print(f"{'policy':>8} {'F1':>6} {'recall':>7} {'prec':>6}")
    lbls = {"D016":"현재(per-frag scalar, thick)", "t1":"t=1 uniform thin", "pervox":"per-voxel medial (수정안)"}
    for p in POL:
        arr = np.array(agg[p]); print(f"{p:>8} {arr[:,0].mean():>6.3f} {arr[:,1].mean():>7.3f} {arr[:,2].mean():>6.3f}   {lbls[p]}")
    d = np.array(agg["D016"])[:,1].mean()
    for p in ["t1", "pervox"]:
        r = np.array(agg[p])[:,1].mean(); print(f"  Δrecall {p} vs D016 = {r-d:+.3f}")
    best = max(POL, key=lambda p: np.array(agg[p])[:,1].mean()); br = np.array(agg[best])[:,1].mean()
    print("  VERDICT:", f"얇은/per-voxel 우세({best} +{br-d:.3f}) -> 라벨이 병목, 재학습 정당화" if br-d > 0.03
          else "두께 이득 미미 -> 라벨이 병목 아님(다른 원인)")
    print("THIN_ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
