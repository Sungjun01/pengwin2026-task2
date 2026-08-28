"""실제 공개 5케이스 프록시-GT 평가 하네스.

dataset_validation 폴더엔 GT가 없음(_data=_output=우리 예측). 대신:
  002/004/005 는 v18 이 리더보드 1.0 만점 → **v18 출력 = 사실상 정답(프록시 GT)**.
  따라서 어떤 후보 출력도 *이 3케이스에서 v18 을 재현하나(=회귀 안 하나)* 를 실제 데이터로 평가.
  003/001 은 v18≠1.0 이라 프록시 불완전 → 구조(조각수)만 참고.

용법: python -m scripts.proxy_gt_eval --cand_dir <후보출력폴더> --cand_pattern '{c}'
  후보 출력은 케이스별 .mha (라벨맵). v18 프록시GT 는 /tmp/real{c}_v18/.../*.mha 에서 자동 로드.
"""
from __future__ import annotations
import argparse, sys, tempfile, os
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.pengwin_eval import evaluate_case

PROXY_CASES = ["002", "004", "005"]   # v18=1.0 → 프록시 GT 신뢰
STRUCT_CASES = ["001", "003"]          # v18≠1.0 → 구조 참고만


def v18_path(c):
    d = f"/tmp/real{c}_v18/out/images/peripelvic-fracture-ct-segmentation"
    g = list(Path(d).glob("*.mha")) if Path(d).exists() else []
    return str(g[0]) if g else None


def nfrag(p):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(int)
    out = {}
    for nm, (lo, hi) in [("sac", (1, 50)), ("LH", (51, 100)), ("RH", (101, 150)), ("fem", (151, 200))]:
        n = len([v for v in np.unique(a) if lo <= v <= hi])
        if n:
            out[nm] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand_dir", required=True, help="후보 출력 .mha 들이 있는 폴더")
    ap.add_argument("--cand_glob", default="{c}", help="케이스 c 의 후보파일명 패턴(확장자 제외 substring)")
    args = ap.parse_args()

    print("=== 프록시-GT 평가 (002/004/005: v18=정답) ===")
    print(f"{'case':5s} {'ins_f1 vs v18':>13s}  {'후보 조각':>20s}  {'v18 조각':>20s}")
    f1s = []
    for c in PROXY_CASES:
        gt = v18_path(c)
        if gt is None:
            print(f"{c:5s}  v18 프록시GT 없음 (real{c}_v18 먼저 실행)"); continue
        # 후보 파일 찾기
        cand = None
        for f in Path(args.cand_dir).glob("*.mha"):
            if c in f.stem:
                cand = f; break
        if cand is None:
            print(f"{c:5s}  후보 출력 없음"); continue
        r = evaluate_case(str(cand), gt, min_cc_mm3=1000)
        f1 = r.get("RQ_F", float("nan"))
        f1s.append(f1)
        flag = " ✓" if f1 > 0.99 else (" ⚠회귀" if f1 < 0.95 else "")
        print(f"{c:5s} {f1:13.3f}  {str(nfrag(cand)):>20s}  {str(nfrag(gt)):>20s}{flag}")
    if f1s:
        print(f"\n프록시 평균 ins_f1 (vs v18): {np.nanmean(f1s):.4f}  (<1.0 이면 회귀 발생 = 제출 금지)")

    print("\n=== 구조 참고 (001/003: GT없음, 조각수만) ===")
    for c in STRUCT_CASES:
        cand = None
        for f in Path(args.cand_dir).glob("*.mha"):
            if c in f.stem:
                cand = f; break
        v = v18_path(c)
        cs = nfrag(cand) if cand else "?"
        vs = nfrag(v) if v else "?"
        print(f"{c:5s}  후보={cs}  v18={vs}")
    print("PROXY_EVAL_DONE")


if __name__ == "__main__":
    main()
