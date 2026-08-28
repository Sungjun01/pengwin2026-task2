"""BET-1 ORACLE GATE: does long-range signed affinity + Mutex Watershed
SEPARATE adjacent fragments when the SHORT-range seam is invisible (model fails)?

This is a DECODE-CEILING test (zero training): it gives the decode the BEST-CASE
long-range signal and asks whether the decode can exploit it. It does NOT test
learnability (only training answers that). A FAIL here kills the method (like the
Gaussian-Mamba 0.625 ceiling); a PASS justifies committing to training.

Conditions (per sacrum crop of worst under-segmented cases):
  O  oracle, all offsets honest+confident (0.9)         -> sanity: repr+decode works (~1.0 expected)
  S  short-only, honest                                 -> control: ceiling if seam were readable short-range
  A  short-only, SEAM CORRUPTED (model merges at seam)  -> reproduces v18 under-seg (expect FAIL)
  B  short corrupted + LONG-range repulsive honest      -> THE TEST: does long-range mutex re-separate?

Corruption model (realistic):
  same-instance edges (all ranges): attractive, conf 0.9   (model reads "same body" fine)
  DIFFERENT-instance SHORT edges (the invisible seam): model WRONGLY predicts weak-attractive 0.6
  DIFFERENT-instance LONG edges (across fragment bodies): model predicts repulsive conf 0.9 (gross morphology readable)
Mechanism if B works: MWS processes 0.9 long-range repulsive FIRST -> sets mutex between fragment
  components -> the later weak 0.6 seam-merge is BLOCKED -> separation preserved.
Failure mode B reveals: two adjacent fragments with NO long-range edge spanning their cores
  (thin slivers, awkward geometry) -> cannot be separated -> honest "die" signal.
"""
import sys, os, time, argparse
import numpy as np
import SimpleITK as sitk

REPO = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee"
sys.path.insert(0, REPO)
from scripts.affinity_labels import AffinityOffset, build_affinity_targets
from scripts.mutex_watershed_decode import mutex_watershed_decode

GT_DIR = f"{REPO}/predictions/gt_instance"
SACRUM = (1, 50)
CONF_HI, CONF_LO = 0.9, 0.6

SHORT = [AffinityOffset(*o) for o in [(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(4,0,0),(0,4,0),(0,0,4)]]
LONG  = [AffinityOffset(*o) for o in [(8,0,0),(0,8,0),(0,0,8),(16,0,0),(0,16,0),(0,0,16)]]
ALL   = SHORT + LONG
def radius(o): return max(abs(o.dz), abs(o.dy), abs(o.dx))


def greedy_f1(gt, pred, iou_thr=0.5):
    """instance F1/recall/precision via contingency + greedy highest-IoU match."""
    gids = np.array(sorted(int(x) for x in np.unique(gt) if x > 0))
    pids = np.array(sorted(int(x) for x in np.unique(pred) if x > 0))
    G, P = len(gids), len(pids)
    if G == 0:
        return dict(n_gt=0, n_pred=P, f1=1.0 if P == 0 else 0.0, rec=1.0, prec=1.0 if P == 0 else 0.0)
    if P == 0:
        return dict(n_gt=G, n_pred=0, f1=0.0, rec=0.0, prec=0.0)
    gl = np.zeros(int(gt.max()) + 1, np.int64); gl[gids] = np.arange(1, G + 1)
    pl = np.zeros(int(pred.max()) + 1, np.int64); pl[pids] = np.arange(1, P + 1)
    comb = (gl[gt] * (P + 1) + pl[pred]).ravel()
    cont = np.bincount(comb, minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1).astype(np.float64)
    inter = cont[1:, 1:]; gsz = cont[1:, :].sum(1); psz = cont[:, 1:].sum(0)
    union = gsz[:, None] + psz[None, :] - inter
    iou = np.divide(inter, np.maximum(union, 1.0))
    used_g, used_p, tp = set(), set(), 0
    for v, i, j in sorted(((iou[i, j], i, j) for i in range(G) for j in range(P) if iou[i, j] >= iou_thr), reverse=True):
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); tp += 1
    fn = G - tp; fp = P - tp
    rec = tp / (tp + fn) if tp + fn else 1.0
    prec = tp / (tp + fp) if tp + fp else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_gt=G, n_pred=P, f1=f1, rec=rec, prec=prec)


def make_affinities(lbl, offsets, mode):
    """Return (attractive, repulsive) [n_off,Z,Y,X] for a condition.
    same -> attractive 0.9 (all ranges).
    mode 'O': diff -> repulsive 0.9 (honest, all ranges).
    mode 'A': SHORT only; diff(short) -> weak attractive 0.6 (corrupt seam); no repulsive.
    mode 'B': diff SHORT -> weak attractive 0.6 (corrupt); diff LONG -> repulsive 0.9 (honest).
    """
    tg = build_affinity_targets(lbl, offsets=tuple(offsets))
    same = tg.attractive  # 1 where same-instance & valid
    diff = tg.repulsive   # 1 where different-instance & valid
    att = CONF_HI * same.copy()
    rep = np.zeros_like(att)
    for i, o in enumerate(offsets):
        is_long = radius(o) >= 8
        if mode == "O":
            rep[i] = CONF_HI * diff[i]
        elif mode == "S":
            rep[i] = CONF_HI * diff[i]   # short-only honest (offsets already short)
        elif mode == "A":
            att[i] += CONF_LO * diff[i]  # corrupt: seam looks like weak merge
        elif mode == "B":
            if is_long:
                rep[i] = CONF_HI * diff[i]      # long-range honest repulsive
            else:
                att[i] += CONF_LO * diff[i]     # short seam corrupted to weak merge
    return att, rep


def run_case(num, stride=1, maxvox=450000, force_stride=None):
    if force_stride is not None:
        stride = force_stride
    gf = f"{GT_DIR}/PENGWIN_{num}.nii.gz"
    if not os.path.exists(gf):
        return None
    gim = sitk.DICOMOrient(sitk.ReadImage(gf), "LPS")
    gt = sitk.GetArrayFromImage(gim).astype(np.int32)
    sac = ((gt >= SACRUM[0]) & (gt <= SACRUM[1]))
    if sac.sum() == 0:
        return None
    idx = np.where(sac)
    sl = tuple(slice(max(0, idx[d].min() - 2), min(sac.shape[d], idx[d].max() + 3)) for d in range(3))
    lbl = gt[sl].copy(); lbl[~((lbl >= SACRUM[0]) & (lbl <= SACRUM[1]))] = 0
    # relabel sacrum frags 1..k
    ids = sorted(int(v) for v in np.unique(lbl) if v > 0)
    remap = np.zeros(int(lbl.max()) + 1, np.int32)
    for k, v in enumerate(ids, 1): remap[v] = k
    lbl = remap[lbl]
    if stride > 1:
        lbl = lbl[::stride, ::stride, ::stride]
    nv = int((lbl > 0).sum())
    if nv > maxvox and stride == 1:
        return run_case(num, stride=2, maxvox=maxvox)
    fg = lbl > 0
    out = {"case": num, "n_gt": len(ids), "nvox": nv, "stride": stride}
    for cond, offs, mode in [("O", ALL, "O"), ("S", SHORT, "S"), ("A", SHORT, "A"), ("B", ALL, "B")]:
        t = time.time()
        att, rep = make_affinities(lbl, offs, mode)
        pred = mutex_watershed_decode(fg, att, rep, offs, 0.5, 0.5, target_range=(1, 4000))
        sc = greedy_f1(lbl, pred)
        out[cond] = dict(f1=round(sc["f1"], 3), rec=round(sc["rec"], 3), prec=round(sc["prec"], 3),
                         n_pred=sc["n_pred"], t=round(time.time() - t, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=0, help="0=auto, else force")
    a = ap.parse_args()
    fs = a.stride if a.stride > 0 else None
    rows = []
    print(f"{'case':>5} {'nGT':>3} {'nvox':>7} {'st':>2} | {'O.f1':>5} {'S.f1':>5} {'A.f1':>5} {'A.np':>4} | {'B.f1':>5} {'B.rec':>5} {'B.np':>4} {'B.t':>5}", flush=True)
    for num in a.cases:
        r = run_case(num, force_stride=fs)
        if r is None:
            print(f"{num}: no sacrum", flush=True); continue
        rows.append(r)
        print(f"{r['case']:>5} {r['n_gt']:>3} {r['nvox']:>7} {r['stride']:>2} | "
              f"{r['O']['f1']:>5} {r['S']['f1']:>5} {r['A']['f1']:>5} {r['A']['n_pred']:>4} | "
              f"{r['B']['f1']:>5} {r['B']['rec']:>5} {r['B']['n_pred']:>4} {r['B']['t']:>5}", flush=True)
    # O-filter: only trust cases where oracle (all-honest) recovers GT (topology preserved by stride)
    valid = [r for r in rows if r["O"]["f1"] >= 0.85]
    excluded = [r["case"] for r in rows if r["O"]["f1"] < 0.85]
    if excluded:
        print(f"\n  [excluded {len(excluded)} cases with O.f1<0.85 (downsample/topology artifact): {excluded}]")
    if valid:
        agg = lambda c, k: float(np.mean([r[c][k] for r in valid]))
        print("\n===== AGGREGATE (n=%d O-valid sacral cases) =====" % len(valid))
        print(f"  O (oracle all-honest)         f1={agg('O','f1'):.3f}   <- sanity ~1.0 (repr+decode works)")
        print(f"  S (short honest)              f1={agg('S','f1'):.3f}   <- ceiling if seam readable short-range")
        print(f"  A (short, seam corrupted)     f1={agg('A','f1'):.3f} rec={agg('A','rec'):.3f}  <- v18-like FAIL (under-seg)")
        print(f"  B (short corrupt+long honest) f1={agg('B','f1'):.3f} rec={agg('B','rec'):.3f}  <- THE TEST")
        dB = agg('B', 'f1') - agg('A', 'f1')
        dR = agg('B', 'rec') - agg('A', 'rec')
        print(f"\n  long-range RESCUE  B - A = {dB:+.3f} f1   (recall {dR:+.3f})")
        print("  VERDICT:", "LONG-RANGE HELPS (alive -> justify training)" if dB > 0.05 else
              "NO RESCUE (long-range gives ~nothing over short -> DIE)")
    print("AFFINITY_ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
