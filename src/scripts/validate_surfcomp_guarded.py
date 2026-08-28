"""면-완성 + CT-evidence guard 검증 — 058 과분할 교정 + 097/063/075 win 유지?

k=2 고정, guard threshold(thr) 스윕. legacy 대비 9 fold_3 사크럼 RQ.
목표: clean(회귀 0) + mean↑.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.surface_completion_decode import (surface_completion_sacrum, ct_evidence_guard,
                                               rq_sac, lps)

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
CASES = ["097", "063", "058", "006"]  # win / Tversky-broke-now-win / regress / win (핵심 4)
K = 2
THRS = [40.0, 60.0, 80.0, 100.0]


def main():
    legacy = {}; guarded = {t: {} for t in THRS}; raw = {}
    for c in CASES:
        pim, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz"); sp = pim.GetSpacing()
        vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(Path(RAW) / c / "label.mha"); _, ct = lps(Path(RAW) / c / "image.mha"); ct = ct.astype(np.float32)
        leg = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        legacy[c] = rq_sac(gt, leg, spz)
        inst = surface_completion_sacrum(sem, K)
        raw[c] = rq_sac(gt, inst, spz)
        for t in THRS:
            g = ct_evidence_guard(inst, ct, t)
            guarded[t][c] = rq_sac(gt, g, spz)
    print(f"=== 면-완성 k={K} + CT-evidence guard 스윕 (9 fold_3, 사크럼 RQ) ===")
    print(f"{'case':5s} {'legacy':>7s} {'raw(noguard)':>12s} " + " ".join(f"{'thr'+str(int(t)):>7s}" for t in THRS))
    for c in CASES:
        print(f"{c:5s} {legacy[c]:7.3f} {raw[c]:12.3f} " + " ".join(f"{guarded[t][c]:7.3f}" for t in THRS))
    legm = np.nanmean(list(legacy.values()))
    print(f"\n{'mean':5s} {legm:7.3f} {np.nanmean(list(raw.values())):12.3f} " +
          " ".join(f"{np.nanmean(list(guarded[t].values())):7.3f}" for t in THRS))
    print()
    for t in THRS:
        gm = np.nanmean(list(guarded[t].values()))
        reg = [c for c in CASES if guarded[t][c] < legacy[c] - 0.02]
        win = [c for c in CASES if guarded[t][c] > legacy[c] + 0.02]
        clean = "✅CLEAN" if not reg else ""
        print(f"thr={int(t)}: mean Δ={gm-legm:+.4f}  win={win}  regress={reg}  {clean}")
    print("GUARD_SWEEP_DONE")


if __name__ == "__main__":
    main()
