"""v28 Monotone-Safe 디코더 단위테스트 — 합성 케이스로 *구조적 안전속성* 증명 (GPU 불필요).

핵심 주장(v27 참사 차단):
  A. intact femur(1 core CC)로 watershed 과분할 → 1 인스턴스로 병합 (over-split 구출)
  B. 전위 골절(d013가 2 core CC로 봄) → 2 인스턴스 유지 (정밀 분할)
  C. ★전위 골절인데 d013 침묵(core 비어있음)★ → watershed 분할 *유지*(≥2), 절대 1로 봉합 안 함
     (= v27이 004/005에서 터진 바로 그 시나리오. v28은 침묵을 분할유지로 처리)
  D. 양측 femur(분리된 2 덩어리, 각자 core) → 2 인스턴스
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pivot_assembly import monotone_safe_femur_decode

FEMUR = (151, 200)


def n_inst(out):
    return len([v for v in np.unique(out) if FEMUR[0] <= v <= FEMUR[1]])


def two_cubes_with_neck(neck_w=1):
    """d004: 두 정육면체(40 떨어짐)를 얇은 목으로 연결 = 1 connected component (전위골절 모사)."""
    vol = np.zeros((30, 40, 120), dtype=bool)
    vol[8:22, 8:32, 10:45] = True      # 왼쪽 조각
    vol[8:22, 8:32, 75:110] = True     # 오른쪽 조각 (간극 45~75)
    # 얇은 목으로 연결
    cz, cy = 15, 20
    vol[cz-neck_w:cz+neck_w, cy-neck_w:cy+neck_w, 45:75] = True
    return vol


def main():
    vv = 1.0
    results = []

    # --- A. intact femur (긴 단일 덩어리, 1 core CC) → 1 인스턴스 ---
    d004 = np.zeros((30, 40, 120), dtype=bool)
    d004[8:22, 8:32, 10:110] = True          # 긴 단일 femur (watershed가 과분할할 형태)
    core = d004.copy()                        # d013 core = 전체 (1 CC)
    border = np.zeros_like(d004)
    outA = monotone_safe_femur_decode(d004, core, border, vv, target_range=FEMUR, min_volume_mm3=50.0)
    nA = n_inst(outA)
    results.append(("A intact→merge", nA, nA == 1, "1"))

    # --- B. 전위 골절, d013가 2 core CC로 봄 → 2 인스턴스 ---
    d004 = two_cubes_with_neck()
    core = np.zeros_like(d004)
    core[10:20, 10:30, 12:43] = True          # 왼쪽 core
    core[10:20, 10:30, 77:108] = True         # 오른쪽 core (목엔 core 없음 → 2 CC)
    border = np.zeros_like(d004); border[10:20, 10:30, 45:75] = True  # 목에 border
    outB = monotone_safe_femur_decode(d004, core, border, vv, target_range=FEMUR, min_volume_mm3=50.0)
    nB = n_inst(outB)
    results.append(("B displaced(d013 본다)→split", nB, nB == 2, "2"))

    # --- C. ★전위 골절인데 d013 침묵(core 비어있음)★ → watershed 분할 유지(≥2), 1로 봉합 금지 ---
    d004 = two_cubes_with_neck()
    core = np.zeros_like(d004)                 # d013 완전 침묵 (recall-miss 모사)
    border = np.zeros_like(d004)
    outC = monotone_safe_femur_decode(d004, core, border, vv, target_range=FEMUR, min_volume_mm3=50.0)
    nC = n_inst(outC)
    results.append(("C displaced(d013 침묵)→분할유지", nC, nC >= 2, ">=2 (NOT 1)"))

    # --- D. 양측 femur (분리된 2 덩어리, 각자 core) → 2 인스턴스 ---
    d004 = np.zeros((30, 40, 120), dtype=bool)
    d004[8:22, 8:32, 10:50] = True
    d004[8:22, 8:32, 70:110] = True            # 완전 분리 (목 없음 → 2 CC)
    core = d004.copy()
    border = np.zeros_like(d004)
    outD = monotone_safe_femur_decode(d004, core, border, vv, target_range=FEMUR, min_volume_mm3=50.0)
    nD = n_inst(outD)
    results.append(("D bilateral→2", nD, nD == 2, "2"))

    print("\n=== v28 Monotone-Safe 디코더 단위테스트 ===")
    print(f"{'case':36s} {'n_inst':>7s} {'expect':>12s} {'PASS':>5s}")
    allok = True
    for name, n, ok, exp in results:
        allok &= ok
        print(f"{name:36s} {n:>7d} {exp:>12s} {'✅' if ok else '❌':>5s}")
    print(f"\n{'ALL PASS ✅' if allok else 'FAIL ❌'}  "
          f"— 특히 C(침묵→분할유지)가 v27 참사 차단의 핵심 증명")
    print("MONOTONE_TEST_DONE")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
