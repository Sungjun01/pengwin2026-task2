"""production decode baseline 확인: border_core_to_instances(v18 실제) vs assemble legacy(내 검증).

같으면 surfcomp +0.028 이 v18 컨테이너에 전이됨. 다르면 re-baseline 필요.
사크럼 RQ, 9 fold_3 케이스.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.pivot_form_postproc import border_core_to_instances
from scripts.surface_completion_decode import rq_sac, lps

RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
CASES = ["097", "063", "058", "006", "075"]
# D016 7-class: sacrum core=1 border=2
SAC_RANGE = (1, 50)


def main():
    print(f"{'case':5s} {'assemble_leg':>12s} {'bordercore_leg':>14s} {'Δ':>7s}")
    a_all, b_all = [], []
    for c in CASES:
        pim, sem = lps(Path(OOF) / f"PENGWIN_{c}.nii.gz"); sp = pim.GetSpacing()
        vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(Path(RAW) / c / "label.mha")
        a = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        b = border_core_to_instances(sem, 1, 2, SAC_RANGE, vv, min_volume_mm3=500,
                                     form_cfg=None, spacing_zyx=spz)
        ra = rq_sac(gt, a, spz); rb = rq_sac(gt, b, spz)
        a_all.append(ra); b_all.append(rb)
        print(f"{c:5s} {ra:12.3f} {rb:14.3f} {rb-ra:+7.3f}")
    print(f"{'mean':5s} {np.nanmean(a_all):12.3f} {np.nanmean(b_all):14.3f} {np.nanmean(b_all)-np.nanmean(a_all):+7.3f}")
    diff = abs(np.nanmean(b_all) - np.nanmean(a_all))
    print(f"\n판정: 두 legacy decode {'동등 (내 baseline 유효 → surfcomp 전이됨)' if diff < 0.02 else '다름 (re-baseline 필요)'}")
    print("PROD_BASELINE_DONE")


if __name__ == "__main__":
    main()
