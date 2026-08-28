"""atlas ↔ 모델오차 상관 — fracture-prior(heatmap) 효용의 결정적 게이트 (report 42/44 미해결).

질문: 모델이 OOF 에서 *놓친 seam* 들이 fracture atlas 의 *높은(예측가능) 곳* 에 몰리나?
  - 놓친 seam 이 atlas-HIGH 면 → prior 가 "여기 봐라" 라고 모델을 밀 수 있는 자리 (단, 신호 있어야).
  - atlas-LOW(예측불가 위치)면 → prior 무용.
  교차: atlas-HIGH/LOW × HU-가시(>=40)/불가시(<40).
    HIGH+가시 = fracture-prior + recall loss 가 살릴 수 있는 표적.
    HIGH+불가시 = 영상한계(prior 알아도 신호 없음).
    LOW = prior 가 못 도움.

atlas: /tmp/fracture_atlas_hist.npy (60^3, ±150mm, 5mm bin, 사크럼 centroid 기준, zyx).

용법: python -m scripts.oof_atlas_error_correlation
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import generate_binary_structure, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
GTD = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance"
OOF = "/tmp/oof_d016_sac"
ATLAS = "/tmp/fracture_atlas_hist.npy"
R, BIN, N = 150.0, 5.0, 60
LO, HI = 1, 50
CLEANUP = 1000.0


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac_inst(lbl):
    return {int(v): (lbl == v) for v in np.unique(lbl) if LO <= v <= HI}


def atlas_value(seam_centroid_zyx, sac_centroid_zyx, sp, atlas, thr_high):
    off_mm = (np.array(seam_centroid_zyx) - np.array(sac_centroid_zyx)) * sp
    idx = ((off_mm + R) / BIN).astype(int)
    if np.any(idx < 0) or np.any(idx >= N):
        return None, False
    v = float(atlas[idx[0], idx[1], idx[2]])
    return v, (v >= thr_high)


def main():
    atlas = np.load(ATLAS)
    occ = atlas[atlas > 0]
    thr_high = np.percentile(occ, 90)  # top-10% bin 임계 (occupied bins 기준)
    print(f"atlas: occupied bins={len(occ)}, top-10% thr={thr_high:.4f}, max={atlas.max():.3f}")

    frac = [l.strip() for l in open("/tmp/frac_sacrum_cases.txt") if l.strip()]
    rows = []  # (atlas_v, is_high, contrast)
    for c in frac:
        cid = f"PENGWIN_{c}"
        pp = Path(OOF) / f"{cid}.nii.gz"; gp = Path(GTD) / f"{cid}.nii.gz"; ip = Path(RAW) / c / "image.mha"
        if not (pp.exists() and gp.exists() and ip.exists()):
            continue
        pim, sem = lps(pp); sp_xyz = pim.GetSpacing(); vv = float(np.prod(sp_xyz)); spz = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
        _, gt = lps(gp); _, ct = lps(ip); ct = ct.astype(np.float32)
        gt = np.where((gt >= 1) & (gt <= 150), gt, 0)
        inst = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        inst = remove_small_components(inst.astype(np.uint16), spz, CLEANUP)
        gi = sac_inst(gt); pi = sac_inst(inst)
        if len(gi) < 2:
            continue
        sac_all = (gt >= LO) & (gt <= HI)
        if not sac_all.any():
            continue
        sac_cen = ndi.center_of_mass(sac_all)  # zyx voxel
        sp_arr = np.array(spz)
        for pid, pmask in pi.items():
            covered = [gid for gid, gm in gi.items() if (pmask & gm).sum() > 0.2 * gm.sum()]
            for i in range(len(covered)):
                for j in range(i + 1, len(covered)):
                    gA, gB = gi[covered[i]], gi[covered[j]]
                    seam = (binary_dilation(gA, ST, 1) & gB) | (binary_dilation(gB, ST, 1) & gA)
                    if seam.sum() < 3:
                        continue
                    scz = ndi.center_of_mass(seam)  # zyx voxel
                    av, is_high = atlas_value(scz, sac_cen, sp_arr, atlas, thr_high)
                    if av is None:
                        continue
                    bone = ct[(gA | gB)]; bone_med = float(np.median(bone[bone >= 0])) if (bone >= 0).any() else 200.0
                    contrast = bone_med - float(np.median(ct[seam]))
                    rows.append((av, is_high, contrast))

    if not rows:
        print("no missed seams found"); return
    avs = np.array([r[0] for r in rows]); highs = np.array([r[1] for r in rows]); cons = np.array([r[2] for r in rows])
    n = len(rows)
    print(f"\n=== OOF 놓친 seam {n}개의 atlas 위치 ===")
    print(f"atlas-HIGH(top10%)에 속한 비율: {100*highs.mean():.0f}%  (무작위면 ~10%)")
    print(f"  → {'몰림(prior가 오차 위치를 겨냥)' if highs.mean() > 0.2 else '안 몰림(prior가 오차를 못 겨냥)'}")
    vis = cons >= 40
    print(f"\n=== 교차표 (atlas × HU가시성) ===")
    for hi_lbl, hmask in [("atlas-HIGH", highs), ("atlas-LOW", ~highs)]:
        nh = hmask.sum()
        if nh == 0:
            print(f"{hi_lbl:10s}: 0"); continue
        nv = (hmask & vis).sum(); ni = (hmask & ~vis).sum()
        print(f"{hi_lbl:10s}: {nh:3d}개  | 가시(>=40HU) {nv} ({100*nv/nh:.0f}%) | 불가시(<40) {ni} ({100*ni/nh:.0f}%)")
    target = (highs & vis).sum()
    print(f"\n>>> fracture-prior 표적(atlas-HIGH AND 가시>=40HU) = {target}/{n} = {100*target/n:.0f}%")
    print(f"    (이게 prior+recall loss로 실제 살릴 수 있는 상한. 나머지는 영상한계 or 위치예측불가.)")
    print(f"판정: {'fracture-prior 추구 정당 (>=20%)' if target/n >= 0.2 else 'fracture-prior 효용 낮음'}")
    print("ATLAS_ERROR_DONE")


if __name__ == "__main__":
    main()
