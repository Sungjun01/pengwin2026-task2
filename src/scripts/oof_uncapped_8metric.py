"""OOF 8-metric baseline — D016(production trainer)의 uncapped validation 예측을 8지표로 채점.

목적:
  1. 한 번도 안 잰 **anatomy 3지표**(IoU_A / HD95_A / ASSD_A)를 OOF로 측정 (랭크의 3/8 사각지대).
  2. nnUNet validation 예측은 기본값(TTA + overlap 0.5)으로 저장됨 = **캡 안 씌운 천장** fracture 5지표.
  3. 이후 A/B/C 개선을 잴 **OOF baseline** 고정.

orientation trap 방지: pred·GT 둘 다 lps()로 LPS 정합 후 array-레벨 eval 직접 호출
(evaluate_case 는 reorient 안 함 → raw GT(mixed orient)와 비교 시 위험).

용법:
  python3 -m scripts.oof_uncapped_8metric            # 전체 ~190 OOF
  OOF_LIMIT=2 python3 -m scripts.oof_uncapped_8metric  # 스모크(앞 2케이스)
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from scripts.gmm_sliver_merge import lps
from scripts.pivot_assembly import merge_sliver_fragments, AssemblyConfig
from evaluation.pengwin_eval import evaluate_f_level, evaluate_a_level, remove_small_components

# v18 production PIVOT-Assembly config (Dockerfile.v18 ENV)
PROD_CFG = AssemblyConfig(
    max_sliver_volume_mm3=2000.0, min_volume_ratio=5.0,
    dilation_iters=1, max_centroid_distance_mm=30.0,
)

RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
RES = Path(
    "/home/schoaq/taehee/nnUNet_data/results/Dataset016_PENGWIN_BorderCore_MedialOrient/"
    "nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres"
)
MIN_CC = 1000.0      # 공식 평가 small-CC cleanup
MODE = "legacy"      # validate_tversky_oof 와 동일한 base decode
KEYS = ["IoU_A", "HD95_A_mm", "ASSD_A_mm", "IoU_F", "HD95_F_mm", "ASSD_F_mm", "LDSC_F", "RQ_F"]


def _aggregate_and_report(per_case, skipped, out_path, tag):
    agg = {}
    for k in KEYS:
        vals = [c[k] for c in per_case.values() if not np.isnan(c.get(k, float("nan")))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    # outlier rate: fracture HD95 > 10mm (badly-placed fragment)
    f_hd95 = [c["HD95_F_mm"] for c in per_case.values() if not np.isnan(c.get("HD95_F_mm", float("nan")))]
    outlier_rate = float(np.mean([h > 10.0 for h in f_hd95])) if f_hd95 else float("nan")
    out = {
        "n_cases": len(per_case), "n_skipped": len(skipped), "skipped": skipped,
        "mode": MODE, "min_cc_mm3": MIN_CC, "outlier_rate_HD95_F_gt10": outlier_rate,
        "trainer": "nnUNetTrainerBorderWeightedAseg_w40 (production, TTA+overlap default)",
        "aggregate": agg, "per_case": per_case,
    }
    Path("logs").mkdir(exist_ok=True)
    with open(out_path, "w") as fjs:
        json.dump(out, fjs, indent=2)
    print(f"\n=== OOF 8-METRIC [{tag}] (uncapped TTA+overlap, n={len(per_case)}, skipped={len(skipped)}) ===")
    print(f"ANATOMY  : IoU_A={agg['IoU_A']:.4f}  HD95_A={agg['HD95_A_mm']:.3f}  ASSD_A={agg['ASSD_A_mm']:.3f}")
    print(f"FRACTURE : IoU_F={agg['IoU_F']:.4f}  HD95_F={agg['HD95_F_mm']:.3f}  ASSD_F={agg['ASSD_F_mm']:.3f}"
          f"  LDSC_F={agg['LDSC_F']:.4f}  RQ_F={agg['RQ_F']:.4f}")
    print(f"OUTLIER (frac HD95>10mm): {outlier_rate*100:.1f}% of cases" if not np.isnan(outlier_rate) else "")
    print("OOF_8METRIC_DONE")
    return out


def main():
    nworkers = int(os.environ.get("OOF_NWORKERS", "1"))
    worker = int(os.environ.get("OOF_WORKER", "0"))
    limit = int(os.environ.get("OOF_LIMIT", "0"))
    assembly = os.environ.get("OOF_ASSEMBLY", "legacy")   # "legacy" | "production"

    # merge mode: combine all shards → final
    if os.environ.get("OOF_MERGE"):
        per_case, skipped = {}, []
        for w in range(nworkers):
            sp = Path(f"logs/oof_shard_{w}of{nworkers}_{assembly}.json")
            if not sp.exists():
                print(f"WARN missing shard {sp}"); continue
            d = json.loads(sp.read_text())
            per_case.update(d["per_case"]); skipped.extend(d["skipped"])
        _aggregate_and_report(per_case, skipped, f"logs/oof_uncapped_8metric_{assembly}.json", f"MERGED/{assembly}")
        return

    # build full task list, then take this worker's stride
    tasks = []
    for fold in range(5):
        vdir = RES / f"fold_{fold}" / "validation"
        if not vdir.exists():
            continue
        for pf in sorted(vdir.glob("PENGWIN_*.nii.gz")):
            tasks.append(pf)
    tasks = [pf for i, pf in enumerate(tasks) if i % nworkers == worker]
    if limit:
        tasks = tasks[:limit]

    per_case: dict[str, dict] = {}
    skipped = []
    for pf in tasks:
        cid = pf.name.split(".")[0]
        num = cid.split("_")[1]
        gp = RAW / num / "label.mha"
        if not gp.exists():
            skipped.append((cid, "no GT")); continue
        pim, sem = lps(str(pf))
        sp = pim.GetSpacing()
        spz = (sp[2], sp[1], sp[0]); vv = float(np.prod(sp))
        inst = assemble_pelvic_instances_mode(sem, vv, mode=MODE, spacing_zyx=spz, min_volume_mm3=500)
        if assembly == "production":
            inst = merge_sliver_fragments(inst, voxel_volume_mm3=vv, cfg=PROD_CFG, spacing_zyx=spz)
        _, gt = lps(str(gp))
        if inst.shape != gt.shape:
            skipped.append((cid, f"shape {inst.shape} vs gt {gt.shape}")); continue
        inst = remove_small_components(inst, spz, MIN_CC)
        gt2 = remove_small_components(gt, spz, MIN_CC)
        f = evaluate_f_level(gt2, inst, spz)
        a = evaluate_a_level(gt2, inst, spz)
        per_case[cid] = {**f, **a}
        print(f"[w{worker}] {cid} | A IoU={a['IoU_A']:.3f} HD95={a['HD95_A_mm']:.2f}"
              f" | F IoU={f['IoU_F']:.3f} HD95={f['HD95_F_mm']:.2f} LDSC={f['LDSC_F']:.3f} RQ={f['RQ_F']:.3f}",
              flush=True)

    if nworkers > 1:
        out_path = f"logs/oof_shard_{worker}of{nworkers}_{assembly}.json"
        out = {"n_cases": len(per_case), "skipped": skipped, "per_case": per_case}
        Path("logs").mkdir(exist_ok=True)
        Path(out_path).write_text(json.dumps(out))
        print(f"[w{worker}] shard done n={len(per_case)} → {out_path}")
    else:
        _aggregate_and_report(per_case, skipped, f"logs/oof_uncapped_8metric_{assembly}.json", f"FULL/{assembly}")


if __name__ == "__main__":
    main()
