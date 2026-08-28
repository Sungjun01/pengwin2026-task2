"""(A) 전파-완성 vs v24 면-완성 vs legacy — OOF 9 사크럼-골절, 사크럼 RQ.

질문: CT-유도(균열 경로) 완성이 형태학 closing(v24)보다 나은가?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pivot_form_postproc import border_core_to_instances
from scripts.surface_completion_decode import (surface_completion_sacrum,
                                               propagation_completion_sacrum, rq_sac, lps)

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
CASES = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]
PROP_CFGS = [("prop w0.5/d1.0", dict(k=2, w_ct=0.5, w_dist=1.0)),
             ("prop w0/d1.0(pureDist)", dict(k=2, w_ct=0.0, w_dist=1.0)),
             ("prop w0.3/d1.5", dict(k=2, w_ct=0.3, w_dist=1.5)),
             ("prop w0.7/d1.5", dict(k=2, w_ct=0.7, w_dist=1.5))]


def main():
    cols = ["legacy", "surfcomp(v24)"] + [n for n, _ in PROP_CFGS]
    data = {c: {} for c in CASES}
    for c in CASES:
        pim, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz"); sp = pim.GetSpacing()
        vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(Path(RAW) / c / "label.mha"); _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
        leg = border_core_to_instances(sem, 1, 2, (1, 50), vv, min_volume_mm3=500, form_cfg=None, spacing_zyx=spz)
        data[c]["legacy"] = rq_sac(gt, leg, spz)
        data[c]["surfcomp(v24)"] = rq_sac(gt, surface_completion_sacrum(sem, 2), spz)
        for nm, kw in PROP_CFGS:
            data[c][nm] = rq_sac(gt, propagation_completion_sacrum(sem, ct, **kw), spz)
    print(f"=== 전파-완성 vs v24 (사크럼 RQ, 9 OOF) ===")
    print(f"{'case':5s} " + " ".join(f"{c[:13]:>14s}" for c in cols))
    for c in CASES:
        print(f"{c:5s} " + " ".join(f"{data[c][col]:14.3f}" for col in cols))
    print(f"{'mean':5s} " + " ".join(f"{np.nanmean([data[c][col] for c in CASES]):14.3f}" for col in cols))
    legm = np.nanmean([data[c]["legacy"] for c in CASES])
    scm = np.nanmean([data[c]["surfcomp(v24)"] for c in CASES])
    print(f"\nlegacy={legm:.4f}  surfcomp(v24)={scm:.4f}")
    for col in cols[2:]:
        m = np.nanmean([data[c][col] for c in CASES])
        reg_leg = [c for c in CASES if data[c][col] < data[c]["legacy"] - 0.02]
        vs_sc = [c for c in CASES if data[c][col] > data[c]["surfcomp(v24)"] + 0.02]
        worse_sc = [c for c in CASES if data[c][col] < data[c]["surfcomp(v24)"] - 0.02]
        print(f"{col}: mean={m:.4f} (Δlegacy={m-legm:+.4f}, Δsurfcomp={m-scm:+.4f}) "
              f"regress_vs_legacy={reg_leg} better_than_surfcomp={vs_sc} worse={worse_sc}")
    print("PROPAGATION_DONE")


if __name__ == "__main__":
    main()
