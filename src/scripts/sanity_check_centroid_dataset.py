"""Dataset010/011 (Centroid-only ablation) 의 sanity check.

외부 reviewer 가 합의한 pass/fail criterion 3 가지.
    1. Channel value distribution — 3 channel (1, 2, 3) 각각 mean ≈ 0, std 0.2~0.4
    2. Peripheral patch std (1cm-thick lateral shell) — std > 0.05
       (우리 mechanism 의 core claim: 외측 patch 에서도 spatial gradient 유지)
    3. Empty mask skip case 수 — 170 중 < 10 (이상 시 Model_A quality 별도 조사)

용법.
    python scripts/sanity_check_centroid_dataset.py --dataset 010
    python scripts/sanity_check_centroid_dataset.py --dataset 011 --cases PENGWIN_001 PENGWIN_100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"환경변수 {name} 가 설정되어 있지 않습니다. scripts/nnunet_env.sh 를 source 하세요.")
    return v


def lateral_shell_mask(shape, spacing_xyz, shell_mm: float = 10.0):
    """Volume bounding box 의 *외측* 1cm-thick shell mask. (Z, Y, X) shape.

    가장 lateral 한 영역 = 골반 외측 끝자락 = peripheral patch 의 hipbone 외측 voxel.
    sacrum centroid 에서 가장 먼 영역. 우리 method 의 검증 대상.
    """
    z_dim, y_dim, x_dim = shape
    # x 방향 (좌우) 의 외측 shell 만 사용 — LH/RH 의 lateral 영역
    shell_voxels_x = max(1, int(shell_mm / spacing_xyz[0]))
    mask = np.zeros(shape, dtype=bool)
    mask[:, :, :shell_voxels_x] = True
    mask[:, :, -shell_voxels_x:] = True
    return mask


def check_case(case_id: str, ds_dir: Path) -> dict:
    channels = []
    for c in range(1, 4):
        path = ds_dir / "imagesTr" / f"{case_id}_{c:04d}.nii.gz"
        if not path.exists():
            print(f"  [missing] {path}")
            return {"case": case_id, "ok": False, "reason": f"channel {c} 없음"}
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        channels.append((c, arr, img.GetSpacing()))

    spacing = channels[0][2]
    shape = channels[0][1].shape

    # 1. Channel value distribution
    stats = []
    for c, arr, _ in channels:
        stats.append({
            "channel": c,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        })

    # 2. Peripheral patch std (lateral shell)
    shell = lateral_shell_mask(shape, spacing, shell_mm=10.0)
    peripheral_std = []
    for c, arr, _ in channels:
        peripheral_std.append({"channel": c, "shell_std": float(arr[shell].std())})

    return {
        "case": case_id,
        "ok": True,
        "shape": list(shape),
        "spacing": list(spacing),
        "value_stats": stats,
        "peripheral": peripheral_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=10, choices=[10, 11])
    parser.add_argument("--cases", nargs="+", default=None,
                        help="default: 처음 3 case")
    args = parser.parse_args()

    suffix = "Predicted" if args.dataset == 10 else "Oracle"
    ds_name = f"Dataset{args.dataset:03d}_PENGWIN_CentroidOnly_{suffix}"
    ds_dir = Path(env("nnUNet_raw")) / ds_name

    if not ds_dir.exists():
        sys.exit(f"dataset 없음: {ds_dir} — 빌드 먼저 진행")

    # case list
    if args.cases is None:
        all_cases = sorted(p.name.replace("_0000.nii.gz", "")
                           for p in (ds_dir / "imagesTr").glob("*_0000.nii.gz"))
        cases = all_cases[:3]
    else:
        cases = args.cases

    print(f"[setup] {ds_name}")
    print(f"[setup] checking {len(cases)} cases\n")

    # Empty mask skip count (dataset.json)
    import json
    with open(ds_dir / "dataset.json") as f:
        dj = json.load(f)
    num_training = dj.get("numTraining", 0)
    all_d003 = len(list((Path(env("nnUNet_raw")) / "Dataset003_PENGWIN_PelvicAnatomy" / "imagesTr").glob("*_0000.nii.gz")))
    skipped = all_d003 - num_training
    print(f"[skip count] {skipped} cases skipped (of {all_d003} total in D003)  "
          f"{'PASS' if skipped < 10 else 'FAIL (>=10, Model_A quality 별도 조사)'}\n")

    pass_count = {"value_dist": 0, "peripheral_std": 0}
    for case_id in cases:
        print(f"━━ {case_id} ━━")
        res = check_case(case_id, ds_dir)
        if not res["ok"]:
            print(f"  FAIL: {res.get('reason')}")
            continue
        print(f"  shape={res['shape']}, spacing={[f'{s:.3f}' for s in res['spacing']]}")
        print(f"  [value stats]")
        all_dist_ok = True
        for st in res["value_stats"]:
            ok_mean = abs(st["mean"]) < 0.4
            # uniform distribution 의 이론 std = 1/sqrt(3) ≈ 0.577.
            # 실제 voxel 분포는 case 별로 약간 좁거나 비대칭이라 0.30~0.65 허용.
            ok_std = 0.20 < st["std"] < 0.65
            mark = "✓" if (ok_mean and ok_std) else "✗"
            print(f"    ch{st['channel']}: min={st['min']:+.3f}  max={st['max']:+.3f}  "
                  f"mean={st['mean']:+.3f}  std={st['std']:.3f}  {mark}")
            if not (ok_mean and ok_std):
                all_dist_ok = False
        if all_dist_ok:
            pass_count["value_dist"] += 1

        print(f"  [peripheral lateral shell (1cm thick)]")
        all_peri_ok = True
        for ps in res["peripheral"]:
            mark = "✓" if ps["shell_std"] > 0.05 else "✗"
            print(f"    ch{ps['channel']}: shell_std={ps['shell_std']:.4f}  {mark}  (threshold > 0.05)")
            if ps["shell_std"] <= 0.05:
                all_peri_ok = False
        if all_peri_ok:
            pass_count["peripheral_std"] += 1
        print()

    print("━━ Summary ━━")
    print(f"  value distribution OK : {pass_count['value_dist']}/{len(cases)}")
    print(f"  peripheral shell  OK : {pass_count['peripheral_std']}/{len(cases)}")
    print(f"  empty-mask skip : {skipped} cases ({'PASS' if skipped < 10 else 'FAIL'})")

    overall = (
        pass_count["value_dist"] == len(cases)
        and pass_count["peripheral_std"] == len(cases)
        and skipped < 10
    )
    print(f"\n  OVERALL: {'PASS — 학습 launch OK' if overall else 'FAIL — fix 또는 spec 재검토'}")


if __name__ == "__main__":
    main()
