"""v27 (d013 femur decode) proxy-GT 검증 — intact femur(004/005) 회귀 안 하나.

dataset_validation의 004/005(femur), 002(pelvic) input을 v27 설정(d013 femur +
chunk-merge off)으로 추론 → v18 출력(_output.mha = 리더보드 1.0 프록시 정답)과 비교.
d013가 intact femur 인스턴스를 보존하면(RQ~1.0) → 빌드+제출 안전.

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N python3 -m scripts.proxy_gt_v27
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# v27 femur 설정 (d013 + chunk off) — import 전에 env 세팅
os.environ.setdefault("PIVOT_PELVIC_DATASET", "Dataset016_PENGWIN_BorderCore_MedialOrient")
os.environ["PIVOT_FEMUR_HYBRID"] = "1"          # D004 + D013 둘 다 로드
os.environ["PIVOT_FEMUR_GLOBAL_CC"] = "1"       # v27-True: 글로벌 CC(D004) + 골절분할(D013)
os.environ["PIVOT_ASSEMBLY"] = "1"
os.environ["PIVOT_ASSEMBLY_FEMUR_CHUNKS"] = "0"
os.environ["PIVOT_ASSEMBLY_MAX_SLIVER_MM3"] = "2000"
os.environ["PIVOT_ASSEMBLY_MIN_VOL_RATIO"] = "5"
os.environ["PIVOT_ASSEMBLY_DILATION_ITERS"] = "1"
os.environ["PIVOT_ASSEMBLY_MAX_DIST_MM"] = "30"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_v7_d015 import run_pipeline_v7
from evaluation.pengwin_eval import evaluate_case

RES = Path("/home/schoaq/taehee/nnUNet_data/results")
DV = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/progress/dataset_validation")
CASES = ["002", "004", "005"]


def femur_inst(arr):
    return sorted(int(v) for v in np.unique(arr) if 151 <= v <= 200)


def all_inst(arr):
    return sorted(int(v) for v in np.unique(arr) if v > 0)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="proxyv27_"))
    print("[proxy-GT v27] d013 femur + chunk-off vs v18 출력(프록시 정답)\n", flush=True)
    for cid in CASES:
        inp = DV / f"{cid}_input.mha"
        gt18 = DV / f"{cid}_output.mha"   # v18 출력 = 프록시 정답
        if not inp.exists():
            print(f"{cid}: input 없음 skip", flush=True); continue
        out = tmp / f"v27_{cid}.mha"
        run_pipeline_v7(input_ct=inp, output_path=out, nnunet_results=RES,
                        tile_step_size=1.0, use_pivot_form=False, min_volume_mm3=500.0)
        v27 = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
        v18 = sitk.GetArrayFromImage(sitk.ReadImage(str(gt18)))
        # 인스턴스 비교
        f27, f18 = femur_inst(v27), femur_inst(v18)
        # RQ vs 프록시 정답 (v18 = GT)
        r = evaluate_case(str(out), str(gt18))
        same = "SAME" if all_inst(v27) == all_inst(v18) else "DIFFER"
        print(f"case {cid}: v18_femur={f18}(n={len(f18)})  v27_femur={f27}(n={len(f27)})  "
              f"RQ_F={r.get('RQ_F', float('nan')):.3f} HD95_F={r.get('HD95_F_mm', float('nan')):.2f}  "
              f"all_inst:{same}", flush=True)
    print("\nPROXYV27_DONE", flush=True)


if __name__ == "__main__":
    main()
