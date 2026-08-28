"""#1 GMM CT-증거 기반 sliver 병합 (학습 0, 사용자 제안).

원리: 두 인접 조각의 *접촉 seam 영역* CT HU 분포에 2-component GMM 피팅.
  - 진짜 골절: 피질골(고밀도) + 골절간극(저밀도) → 두 가우시안 잘 분리(bimodal) → 유지
  - 헛분할(solid bone 과분할): 단일 뼈조직 → 두 가우시안 거의 겹침(unimodal) → 병합
판정: |μ_high - μ_low| < SEP_HU 면 "영상학적 증거 불충분" → 강제 병합.

함정 회피(핵심): 경계 voxel 은 피질골(고밀도)이라, seam 을 *양쪽으로 두껍게 dilate* 해
간극 저밀도 voxel 까지 포함해 샘플링. 안 그러면 둘 다 고밀도로 잡혀 GMM 무의미.

검증: 사크럼 골절 스코어보드(GT) — (a) 헛조각 제거로 RQ↑ (b) 진짜 골절 안 합침.
용법: python -m scripts.gmm_sliver_merge --pred_dir /tmp/pred_d016_frac
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components
from evaluation.matching import match_fragments
from evaluation.metrics import iou, recognition_quality

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
CLEANUP = 1000.0
SEAM_DILATE = 2     # seam 양쪽 dilation iter (간극 포함 위해 두껍게)
SEP_HU = 40.0       # GMM 두 평균 차 < 이 값 → 골절증거 불충분 → 병합
MIN_LOW_FRAC = 0.10 # 저밀도 component 최소 비중 (간극이 의미있게 존재해야 골절)


def has_fracture_evidence(ct, seam_mask):
    """seam 영역 HU 에 2-comp GMM → 골절(저밀도 간극) 증거 있나."""
    hu = ct[seam_mask].reshape(-1, 1)
    if hu.shape[0] < 30:
        return True  # 샘플 부족 → 보수적으로 유지
    hu = np.clip(hu, -200, 1500)
    try:
        g = GaussianMixture(n_components=2, covariance_type="full", random_state=0, n_init=1).fit(hu)
    except Exception:
        return True
    mu = g.means_.flatten(); w = g.weights_.flatten()
    lo_i = int(np.argmin(mu)); hi_i = int(np.argmax(mu))
    sep = mu[hi_i] - mu[lo_i]
    low_frac = w[lo_i]
    # 골절증거 = 두 평균 충분히 분리 AND 저밀도 component 의미있는 비중
    return (sep >= SEP_HU) and (low_frac >= MIN_LOW_FRAC)


def gmm_merge(inst, ct, anat_ranges=((1, 50), (51, 100), (101, 150))):
    """인접 같은-anatomy 조각쌍에 GMM 증거검사 → 증거없으면 병합."""
    out = inst.copy()
    for lo, hi in anat_ranges:
        ids = [int(v) for v in np.unique(out) if lo <= v <= hi]
        if len(ids) < 2:
            continue
        merged = True
        while merged:
            merged = False
            ids = [int(v) for v in np.unique(out) if lo <= v <= hi]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    ma, mb = (out == a), (out == b)
                    contact = binary_dilation(ma, ST, 1) & mb
                    if contact.sum() < 10:
                        continue
                    # seam = 양쪽 두껍게 dilate (간극 저밀도 포함)
                    seam = binary_dilation(ma, ST, SEAM_DILATE) & binary_dilation(mb, ST, SEAM_DILATE)
                    seam &= (ma | mb | binary_dilation(ma | mb, ST, 1))
                    if not has_fracture_evidence(ct, seam):
                        # 증거 불충분 → 작은 쪽을 큰 쪽에 병합
                        small, big = (a, b) if ma.sum() < mb.sum() else (b, a)
                        out[out == small] = big
                        merged = True
                        break
                if merged:
                    break
    return out


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def rq_sac(gt, pred, spz, lo=1, hi=50):
    g = remove_small_components(np.where((gt >= lo) & (gt <= hi), gt, 0).astype(np.uint16), spz, CLEANUP)
    p = remove_small_components(np.where((pred >= lo) & (pred <= hi), pred, 0).astype(np.uint16), spz, CLEANUP)
    m = match_fragments(g, p, "hungarian")
    if not m:
        return float("nan")
    tp = fn = 0; mp = set()
    for gi, pi in m.items():
        if pi is None: fn += 1; continue
        if iou(p == pi, g == gi) >= 0.5: tp += 1; mp.add(int(pi))
        else: fn += 1
    fp = len({int(x) for x in set(np.unique(p)) - {0}} - mp)
    return recognition_quality(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/tmp/pred_d016_frac")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    preds = sorted(Path(args.pred_dir).glob("PENGWIN_*.nii.gz"))
    if args.limit: preds = preds[: args.limit]
    print(f"[gmm_merge] {len(preds)} cases, SEP_HU={SEP_HU}, seam_dilate={SEAM_DILATE}")
    print(f"{'case':6s} {'사크럼RQ base':>12s} {'+GMM merge':>11s} {'Δ':>6s}  npred base→gmm")
    base_rq, gmm_rq = [], []
    for p in preds:
        cid = p.name.replace("PENGWIN_", "").replace(".nii.gz", "")
        pim, sem = lps(p)
        sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(Path(RAW) / cid / "label.mha")
        _, ct = lps(Path(RAW) / cid / "image.mha"); ct = ct.astype(np.float32)
        inst = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        nb = len([v for v in np.unique(inst) if 1 <= v <= 50])
        inst_g = gmm_merge(inst, ct)
        ng = len([v for v in np.unique(inst_g) if 1 <= v <= 50])
        rb = rq_sac(gt, inst, spz); rg = rq_sac(gt, inst_g, spz)
        base_rq.append(rb); gmm_rq.append(rg)
        flag = " ↑" if rg > rb + 0.01 else (" ↓회귀" if rg < rb - 0.01 else "")
        print(f"{cid:6s} {rb:12.3f} {rg:11.3f} {rg-rb:+6.3f}  {nb}→{ng}{flag}", flush=True)
    print(f"\n평균 사크럼RQ: base {np.nanmean(base_rq):.4f} → +GMM {np.nanmean(gmm_rq):.4f}  (Δ{np.nanmean(gmm_rq)-np.nanmean(base_rq):+.4f})")
    print("판정: Δ>0 + 개별 ↓회귀 없음 → GMM 병합이 헛조각만 제거(진짜 골절 보존) → v18에 안전하게 얹음")
    print("GMM_MERGE_DONE")


if __name__ == "__main__":
    main()
