"""DECODE-CALIBRATION disambiguation: is the affinity net's low MWS_f1 (~0.43) due to
an untuned MWS threshold, or a real learnability ceiling?

Loads the trained sacrum affinity net (ep250), on the SAME held-out val split:
  (1) sweeps MWS repulsive/attractive thresholds -> best achievable MWS_f1
  (2) measures the decisive signal: AUC of predicted P(diff) for LONG edges that truly
      connect ADJACENT fragments (the edges MWS needs) vs within-fragment -> core-sep AUC
  (3) histograms predicted P(diff) for true-same vs true-diff long edges (calibration)
Verdict: if best-threshold MWS_f1 >> 0.43 -> it was decode tuning (affinity alive).
         if best ~ 0.43 AND core-sep AUC ~0.5-0.6 -> real learnability ceiling (dead for this setup).
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, "/tmp")
import affinity_train_standalone as A

device = torch.device("cuda")
lo, hi = A.BONE["sacrum"]

# rebuild data + SAME split (seed 0, val_frac 0.2) as training
import glob
nums = sorted(os.path.basename(f).replace("PENGWIN_", "").replace(".nii.gz", "")
              for f in glob.glob(f"{A.GT_DIR}/PENGWIN_*.nii.gz"))
data = []
for num in nums:
    ct, gt = A.load_crop(num, lo, hi)
    if ct is not None and (gt > 0).sum() > 800 and len(np.unique(gt)) >= 2:
        data.append((num, ct, gt))
rng = np.random.RandomState(0); perm = rng.permutation(len(data))
nval = max(4, int(len(data) * 0.2))
val = [data[i] for i in perm[:nval]]
val = sorted(val, key=lambda t: -len(np.unique(t[2])))
print(f"val={len(val)} cases", flush=True)

net = A.BasicUNet(spatial_dims=3, in_channels=1, out_channels=A.NOFF,
                  features=(32, 32, 64, 128, 256, 32)).to(device)
net.load_state_dict(torch.load("/tmp/aff_learn_sacrum/net_last.pt", map_location=device))
net.eval()

# predict affinity (P(same)) for each val case once
preds = []
with torch.no_grad():
    for num, ct, gt in val[:8]:
        x = torch.from_numpy(ct.astype(np.float32))[None, None].to(device)
        logit = A.sliding_window_inference(x, (96, 128, 128), 2, net, overlap=0.25, mode="gaussian")
        psame = torch.sigmoid(logit)[0].cpu().numpy()
        preds.append((num, gt, psame))
print("predicted affinity for", len(preds), "val cases", flush=True)

# (2)+(3): core-separation AUC — long edges connecting ADJACENT different fragments vs within-fragment
from scipy.ndimage import binary_dilation
same_scores, diff_scores = [], []   # predicted P(diff)=1-P(same)
for num, gt, psame in preds:
    g = gt.astype(np.int32); fg = g > 0
    for i, o in enumerate(A.OFFS):
        if not A.IS_LONG[i]:
            continue
        s = tuple(slice(0, g.shape[d]-o[d]) for d in range(3))
        d2 = tuple(slice(o[d], g.shape[d]) for d in range(3))
        ev = fg[s] & fg[d2]
        if not ev.any():
            continue
        pdiff = (1.0 - psame[i][s][ev])
        same = (g[s] == g[d2])[ev]
        same_scores.append(pdiff[same])
        diff_scores.append(pdiff[~same])
ss = np.concatenate(same_scores); ds = np.concatenate(diff_scores)
def auc(pos, neg):  # pos should score higher
    if len(pos) > 200000: pos = np.random.choice(pos, 200000, replace=False)
    if len(neg) > 200000: neg = np.random.choice(neg, 200000, replace=False)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv, kind="stable"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    n1 = len(pos); n0 = len(neg)
    return (ranks[lab == 1].sum() - n1*(n1+1)/2) / (n1*n0)
print(f"\n=== long-edge P(diff) calibration (all valid long edges) ===", flush=True)
print(f"  true-SAME  median P(diff) = {np.median(ss):.3f}  (want LOW)")
print(f"  true-DIFF  median P(diff) = {np.median(ds):.3f}  (want HIGH)")
print(f"  separation AUC (diff>same) = {auc(ds, ss):.3f}   <- can model tell long diff from same at all?")

# (1) MWS sweep WITH repulsive recalibration (boost) — fixes under-confident P(diff)
print(f"\n=== MWS sweep with repulsive BOOST (instance F1 on {len(preds)} val sacrum crops) ===", flush=True)
def mws_f1_at(rep_thr, boost, att_thr=0.5, stride=2):
    f1s = []
    for num, gt, psame in preds:
        g = gt.astype(np.int32)[::stride, ::stride, ::stride]
        ps = psame[:, ::stride, ::stride, ::stride]
        fg = g > 0
        att = np.zeros((A.NOFF,)+g.shape, np.float32); rep = np.zeros_like(att)
        for i in range(A.NOFF):
            if A.IS_LONG[i]: rep[i] = np.clip((1.0 - ps[i]) * boost, 0.0, 1.0)  # boosted repulsive
            else: att[i] = ps[i]
        offs = [A.AffinityOffset(*o) for o in A.OFFS]
        pred = A.mutex_watershed_decode(fg, att, rep, offs, att_thr, rep_thr, target_range=(1, 4000))
        f1s.append((num, A.greedy_f1(g, pred), len([v for v in np.unique(pred) if v>0]), len(np.unique(g))-1))
    return float(np.mean([x[1] for x in f1s])), f1s

best = (0.0, None)
for boost in [1, 4, 8, 15]:
    for rep_thr in [0.5]:
        t = time.time(); f1, detail = mws_f1_at(rep_thr, boost, 0.5)
        npr = np.mean([d[2] for d in detail]); ngt = np.mean([d[3] for d in detail])
        print(f"  boost={boost:>2} rep_thr={rep_thr}  MWS_f1={f1:.3f}  (avg pred {npr:.1f} vs gt {ngt:.1f} frags) ({time.time()-t:.0f}s)", flush=True)
        if f1 > best[0]: best = (f1, (boost, rep_thr))
print(f"\nBEST MWS_f1 = {best[0]:.3f} at (boost,rep_thr)={best[1]}  (vs default boost=1 = ~0.40; oracle ceiling 0.93)")
print("VERDICT:", "DECODE-CALIBRATION was the issue — affinity ALIVE" if best[0] > 0.65 else
      "PARTIAL: boost helps but still below v18" if best[0] > 0.5 else
      "REAL ceiling — recalibration does not rescue")
print("DECODE_SWEEP_DONE", flush=True)
