"""STANDALONE long-range affinity LEARNABILITY gate (zero nnUNet dependency).

Question: can a model PREDICT long-range repulsive affinity at INVISIBLE seams?
(The oracle gate already proved the DECODE can exploit it: RESCUE +0.678 f1.)

Trains a compact MONAI BasicUNet on raw CT -> 15 affinity channels
(offsets {1,2,4,8,16} x 3 axes), masked BCE, fold_0 held-out for honest measure.
Reports periodically:
  (A) long-range repulsive AUC at SEAM band on val  <- the learnability signal
  (B) (final) MWS-decode 11-metric instance-F1 on val sacrum crops vs GT
A compact net is a learnability FLOOR: if it learns the seam separation, a full
nnUNet certainly will. fold_0 here is the honest GATE; production = fold_all.
"""
import sys, os, time, glob, argparse, json
import numpy as np
import SimpleITK as sitk
import torch, torch.nn as nn, torch.nn.functional as F

REPO = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
sys.path.insert(0, REPO)
from scripts.affinity_loss import build_torch_affinity_targets
from scripts.affinity_labels import AffinityOffset, build_affinity_targets
from scripts.mutex_watershed_decode import mutex_watershed_decode
from scripts.saft_signal_channels import build_signal_channels
from monai.networks.nets import BasicUNet
from monai.inferers import sliding_window_inference

GT_DIR = f"{REPO}/predictions/gt_instance"
RAW = f"{REPO}/raw_data/PENGWIN_train"
CACHE = os.environ.get("AFF_CACHE", "/tmp/aff_cache")
OFFS = [(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(4,0,0),(0,4,0),(0,0,4),
        (8,0,0),(0,8,0),(0,0,8),(16,0,0),(0,16,0),(0,0,16)]
NOFF = len(OFFS)
IS_LONG = np.array([max(o) >= 8 for o in OFFS])
BONE = {"sacrum": (1, 50), "pelvic": (1, 150)}


def ctnorm(ct):
    ct = np.clip(ct, -500, 1500).astype(np.float32)
    return (ct + 500.0) / 2000.0  # ~[0,1]


def build_affinity_input(
    ct: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    signal_channels: bool = False,
) -> np.ndarray:
    parts = [ct.astype(np.float32, copy=False)[None]]
    if signal_channels:
        parts.append(build_signal_channels(ct, spacing_zyx))
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_crop(num, lo, hi, pad=8):
    os.makedirs(CACHE, exist_ok=True)
    cf = f"{CACHE}/{num}_{lo}_{hi}.npz"
    if os.path.exists(cf):
        d = np.load(cf)
        if "spacing_zyx" in d:
            return d["ct"], d["gt"], tuple(float(x) for x in d["spacing_zyx"])
        os.remove(cf)
    gf = f"{GT_DIR}/PENGWIN_{num}.nii.gz"; cfile = f"{RAW}/{num}/image.mha"
    if not (os.path.exists(gf) and os.path.exists(cfile)):
        return None, None, None
    gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
    cim = sitk.DICOMOrient(sitk.ReadImage(cfile), "LPS")
    sx, sy, sz = cim.GetSpacing()
    spacing_zyx = np.asarray((float(sz), float(sy), float(sx)), dtype=np.float32)
    gt = sitk.GetArrayFromImage(gim).astype(np.int32)
    ct = sitk.GetArrayFromImage(cim).astype(np.float32)
    if gt.shape != ct.shape:
        return None, None, None
    m = (gt >= lo) & (gt <= hi)
    if m.sum() == 0:
        return None, None, None
    idx = np.where(m)
    sl = tuple(slice(max(0, idx[d].min()-pad), min(m.shape[d], idx[d].max()+pad+1)) for d in range(3))
    gt = gt[sl].copy(); gt[~((gt >= lo) & (gt <= hi))] = 0
    ids = sorted(int(v) for v in np.unique(gt) if v > 0)
    remap = np.zeros(int(gt.max())+1, np.int32)
    for k, v in enumerate(ids, 1): remap[v] = k
    gt = remap[gt].astype(np.int16)
    ct = ctnorm(ct[sl]).astype(np.float16)
    np.savez_compressed(cf, ct=ct, gt=gt, spacing_zyx=spacing_zyx)
    return ct, gt, tuple(float(x) for x in spacing_zyx)


def weighted_bce(logits, target, valid):
    # base 1; long offsets x2; diff(target0) on long x1.5 extra -> emphasize separation
    w = torch.ones_like(target)
    lo = torch.tensor(IS_LONG, device=target.device).view(1, NOFF, 1, 1, 1)
    w = torch.where(lo.expand_as(w), w * 2.0, w)
    w = torch.where(lo.expand_as(w) & (target < 0.5), w * 1.5, w)
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")
    v = valid
    return loss[v].mean() if v.any() else logits.sum() * 0.0


def fast_auc(scores, labels):
    """AUC that label==1 has HIGHER score. labels in {0,1}."""
    labels = labels.astype(np.int8)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), np.float64); ranks[order] = np.arange(1, len(scores)+1)
    # average ties
    return (ranks[labels == 1].sum() - n1*(n1+1)/2) / (n1*n0)


def seam_auc(net, val, lo, hi, device, max_pairs=300000, band=2, signal_channels=False):
    """On val cases: predicted P(diff) AUC discriminating diff vs same LONG edges,
    restricted to a seam band (near true fragment boundaries). Higher AUC = learnable."""
    net.eval()
    all_s, all_l = [], []   # all long edges; all_s = predicted P(diff)
    sb_s, sb_l = [], []     # seam-band long edges
    with torch.no_grad():
        for num, ct, gt, spacing_zyx in val:
            x = torch.from_numpy(
                build_affinity_input(ct, spacing_zyx, signal_channels=signal_channels)
            )[None].to(device)
            logit = sliding_window_inference(x, (96,128,128), 2, net, overlap=0.25, mode="gaussian")
            psame = torch.sigmoid(logit)[0].cpu().numpy()  # [15,Z,Y,X]
            g = gt.astype(np.int32); fg = g > 0
            # seam band: fg voxels adjacent (within band) to a different label
            from scipy.ndimage import binary_dilation
            boundary = np.zeros_like(fg)
            for o in [(1,0,0),(0,1,0),(0,0,1)]:
                s = tuple(slice(0, g.shape[d]-o[d]) for d in range(3))
                d2 = tuple(slice(o[d], g.shape[d]) for d in range(3))
                diff = fg[s] & fg[d2] & (g[s] != g[d2])
                boundary[s] |= diff; boundary[d2] |= diff
            band_mask = binary_dilation(boundary, iterations=band)
            for i, o in enumerate(OFFS):
                if not IS_LONG[i]:
                    continue
                s = tuple(slice(0, g.shape[d]-o[d]) for d in range(3))
                d2 = tuple(slice(o[d], g.shape[d]) for d in range(3))
                ev = fg[s] & fg[d2]
                if not ev.any():
                    continue
                same = (g[s] == g[d2])[ev]
                pdiff = (1.0 - psame[i][s][ev])   # predicted P(diff)
                lab = (~same).astype(np.int8)      # 1 = truly different (repulsive)
                all_s.append(pdiff); all_l.append(lab)
                bm = band_mask[s][ev]
                if bm.any():
                    sb_s.append(pdiff[bm]); sb_l.append(lab[bm])
    def cat_auc(ss, ll):
        if not ss: return float("nan"), 0
        s = np.concatenate(ss); l = np.concatenate(ll)
        if len(s) > max_pairs:
            idx = np.random.choice(len(s), max_pairs, replace=False); s, l = s[idx], l[idx]
        return fast_auc(s, l), int(l.sum())
    a_all, n_all = cat_auc(all_s, all_l)
    a_sb, n_sb = cat_auc(sb_s, sb_l)
    net.train()
    return a_all, a_sb, n_all, n_sb


def greedy_f1(gt, pred, iou_thr=0.5):
    gids = np.array(sorted(int(x) for x in np.unique(gt) if x > 0))
    pids = np.array(sorted(int(x) for x in np.unique(pred) if x > 0))
    G, P = len(gids), len(pids)
    if G == 0: return 1.0 if P == 0 else 0.0
    if P == 0: return 0.0
    gl = np.zeros(int(gt.max())+1, np.int64); gl[gids] = np.arange(1, G+1)
    pl = np.zeros(int(pred.max())+1, np.int64); pl[pids] = np.arange(1, P+1)
    comb = (gl[gt]*(P+1) + pl[pred]).ravel()
    cont = np.bincount(comb, minlength=(G+1)*(P+1)).reshape(G+1, P+1).astype(np.float64)
    inter = cont[1:,1:]; gsz = cont[1:,:].sum(1); psz = cont[:,1:].sum(0)
    iou = inter / np.maximum(gsz[:,None]+psz[None,:]-inter, 1.0)
    ug, up, tp = set(), set(), 0
    for v,i,j in sorted(((iou[i,j],i,j) for i in range(G) for j in range(P) if iou[i,j]>=iou_thr), reverse=True):
        if i in ug or j in up: continue
        ug.add(i); up.add(j); tp += 1
    rec = tp/G; prec = tp/P; return 2*prec*rec/(prec+rec) if prec+rec else 0.0


def mws_eval(net, val, device, stride=2, ncap=6, signal_channels=False):
    """End-to-end: predict affinity -> MWS -> instance F1 vs GT (sacrum crops)."""
    net.eval(); f1s = []
    with torch.no_grad():
        for num, ct, gt, spacing_zyx in val[:ncap]:
            x = torch.from_numpy(
                build_affinity_input(ct, spacing_zyx, signal_channels=signal_channels)
            )[None].to(device)
            logit = sliding_window_inference(x, (96,128,128), 2, net, overlap=0.25, mode="gaussian")
            psame = torch.sigmoid(logit)[0].cpu().numpy()
            g = gt.astype(np.int32)[::stride,::stride,::stride]
            ps = psame[:, ::stride, ::stride, ::stride]
            fg = g > 0
            att = np.zeros((NOFF,)+g.shape, np.float32); rep = np.zeros_like(att)
            offs = [AffinityOffset(*o) for o in OFFS]
            for i in range(NOFF):
                if IS_LONG[i]: rep[i] = 1.0 - ps[i]   # repulsive = P(diff)
                else: att[i] = ps[i]                  # attractive = P(same)
            pred = mutex_watershed_decode(fg, att, rep, offs, 0.5, 0.5, target_range=(1,4000))
            f1s.append(greedy_f1(g, pred))
    net.train()
    return float(np.mean(f1s)) if f1s else float("nan")


def sample_patch(ct, gt, ps=(96,128,128)):
    sh = gt.shape
    pad = [(0, max(0, ps[d]-sh[d])) for d in range(3)]
    if any(p[1] > 0 for p in pad):
        ct = np.pad(ct, pad); gt = np.pad(gt, pad); sh = gt.shape
    for _ in range(8):
        st = tuple(np.random.randint(0, sh[d]-ps[d]+1) for d in range(3))
        sl = tuple(slice(st[d], st[d]+ps[d]) for d in range(3))
        if (gt[sl] > 0).sum() > 500:
            return ct[sl].astype(np.float32), gt[sl].astype(np.int64)
    return ct[sl].astype(np.float32), gt[sl].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bone", default="sacrum", choices=list(BONE))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--out", default="/tmp/aff_learn")
    ap.add_argument("--signal-channels", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    lo, hi = BONE[a.bone]
    nums = a.cases or sorted(os.path.basename(f).replace("PENGWIN_","").replace(".nii.gz","")
                             for f in glob.glob(f"{GT_DIR}/PENGWIN_*.nii.gz"))
    print(f"[load] {len(nums)} cases, bone={a.bone} ids[{lo},{hi}]", flush=True)
    data = []
    t0 = time.time()
    for num in nums:
        ct, gt, spacing_zyx = load_crop(num, lo, hi)
        if ct is not None and (gt > 0).sum() > 800 and len(np.unique(gt)) >= 2:
            data.append((num, ct, gt, spacing_zyx))
    print(f"[load] usable {len(data)} cases in {time.time()-t0:.0f}s", flush=True)
    rng = np.random.RandomState(0); perm = rng.permutation(len(data))
    nval = max(4, int(len(data)*a.val_frac))
    val = [data[i] for i in perm[:nval]]; train = [data[i] for i in perm[nval:]]
    # prefer multi-fragment val cases for a meaningful separation test
    val = sorted(val, key=lambda t: -len(np.unique(t[2])))
    print(f"[split] train={len(train)} val={len(val)} (val multi-frag counts: "
          f"{[int((np.unique(t[2])>0).sum()) for t in val[:8]]})", flush=True)
    in_channels = 4 if a.signal_channels else 1
    net = BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=NOFF,
                    features=(32,32,64,128,256,32)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    log = []
    for ep in range(1, a.epochs+1):
        net.train(); el = 0.0
        for it in range(a.iters):
            xs, gs = [], []
            for _ in range(a.batch):
                ci = train[np.random.randint(len(train))]
                px, pg = sample_patch(ci[1], ci[2])
                xs.append(build_affinity_input(px, ci[3], signal_channels=a.signal_channels)); gs.append(pg)
            x = torch.from_numpy(np.stack(xs)).to(device)
            g = torch.from_numpy(np.stack(gs)).to(device)
            tgt, valid = build_torch_affinity_targets(g, tuple(OFFS))
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logit = net(x)
                loss = weighted_bce(logit.float(), tgt, valid)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            el += float(loss.detach())
        sched.step()
        if ep % a.eval_every == 0 or ep == 1 or ep == a.epochs:
            a_all, a_sb, n_all, n_sb = seam_auc(net, val, lo, hi, device, signal_channels=a.signal_channels)
            msg = (f"[ep {ep:>3}] loss={el/a.iters:.4f}  longAUC_all={a_all:.3f} "
                   f"longAUC_SEAM={a_sb:.3f} (n_seam_diff={n_sb})")
            if ep == a.epochs or (ep % (a.eval_every*5) == 0):
                f1 = mws_eval(net, val, device, signal_channels=a.signal_channels)
                msg += f"  MWS_f1={f1:.3f}"
            print(msg, flush=True)
            log.append({"ep": ep, "loss": el/a.iters, "longAUC_all": a_all, "longAUC_seam": a_sb})
            json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
            torch.save(net.state_dict(), f"{a.out}/net_last.pt")
    print("AFFINITY_LEARN_DONE", flush=True)


if __name__ == "__main__":
    main()
