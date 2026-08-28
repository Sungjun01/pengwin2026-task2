"""Aggregate the Task2 OOF gate: v5 (axis-flipped) vs v6 (self-correcting) click
postproc on the same v19 instance maps. Reads /tmp/oofgate/*.json."""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

HIGHER = {"IoU_F", "LDSC_F", "RQ_F", "IoU_A"}
LOWER = {"HD95_F_mm", "ASSD_F_mm", "HD95_A_mm", "ASSD_A_mm"}
METRICS = ["RQ_F", "IoU_F", "LDSC_F", "IoU_A", "HD95_F_mm", "ASSD_F_mm", "HD95_A_mm", "ASSD_A_mm"]

PELVIC = {"084", "115", "180", "064", "179", "198"}
FEMUR = {"411", "399", "256", "277", "302", "407"}


def collect(records, case_filter=None):
    """Return {tag: {metric: [values]}} over all (case,strategy) entries."""
    acc = {"v5": {m: [] for m in METRICS}, "v6": {m: [] for m in METRICS}}
    n = {"v5": 0, "v6": 0}
    for rec in records:
        case = rec.get("case")
        if case_filter and case not in case_filter:
            continue
        for strat, entry in rec.get("strategies", {}).items():
            for tag in ("v5", "v6"):
                m = entry.get(tag, {})
                if not isinstance(m, dict) or "error" in m:
                    continue
                n[tag] += 1
                for k in METRICS:
                    v = m.get(k)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        acc[tag][k].append(v)
    return acc, n


def report(title, records, case_filter=None):
    acc, n = collect(records, case_filter)
    print(f"\n===== {title}  (v5 n={n['v5']}, v6 n={n['v6']} case×strategy) =====")
    print(f"{'metric':>11} {'v5':>9} {'v6':>9} {'Δ(v6-v5)':>10}  dir  verdict")
    wins = 0; losses = 0
    for m in METRICS:
        v5v = np.mean(acc["v5"][m]) if acc["v5"][m] else float("nan")
        v6v = np.mean(acc["v6"][m]) if acc["v6"][m] else float("nan")
        d = v6v - v5v
        better_high = m in HIGHER
        good = (d > 1e-6) if better_high else (d < -1e-6)
        neutral = abs(d) <= 1e-6
        if not neutral:
            wins += int(good); losses += int(not good)
        tag = "  =" if neutral else (" ✓" if good else " ✗")
        print(f"{m:>11} {v5v:>9.4f} {v6v:>9.4f} {d:>+10.4f}  {'↑' if better_high else '↓'}  {tag}")
    print(f"  -> v6 better on {wins} metrics, worse on {losses}")
    return acc, n


def main():
    files = sys.argv[1:] or sorted(glob.glob("/tmp/oofgate/*.json"))
    records = []
    for f in files:
        try:
            records.extend(json.load(open(f)))
        except Exception as e:
            print(f"[skip] {f}: {e}")
    done = [r for r in records if "strategies" in r]
    print(f"loaded {len(records)} case records ({len(done)} with strategies) from {len(files)} files")
    errs = [r["case"] for r in records if "error" in r]
    if errs:
        print(f"[case-level errors] {errs}")
    report("ALL CASES", records)
    report("PELVIC", records, PELVIC)
    report("FEMUR", records, FEMUR)


if __name__ == "__main__":
    main()
