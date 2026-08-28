"""놓친 seam 의 '부분 가시성/anchor' + 면 방향 검증 — 다중뷰/면-완성 레버 타당성.

질문(팀원 통찰): 놓친 22HU seam 이 사실 *부분적으로는 또렷*한가(피질부 anchor)?
  그렇다면 면-연속성/완성으로 흐린 구간을 이을 수 있음(per-voxel 검출론 못 함).
  + seam 면이 axial 평면에 평행하면 axial 검출이 불리(직교 뷰가 유리) = 다중뷰 이점.

각 OOF 케이스에서 D016 이 병합한 GT 사크럼 fragment 쌍(=놓친 seam)마다:
  (a) 접촉면 voxel 별 대조(=인접 뼈 median - seam HU) 분포 → max(anchor), 보이는 비율(>=40HU)
  (b) 접촉면 법선 방향(PCA 최소분산축) → axial(z)축과의 각 → 평행(=axial서 불리)인가
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.form_learn_postproc import assemble_pelvic_instances_mode
from evaluation.pengwin_eval import remove_small_components

ST = generate_binary_structure(3, 3)
RAW = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train"
OOF = "/tmp/oof_d016_sac"
LO, HI = 1, 50


def lps(p):
    im = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    return im, sitk.GetArrayFromImage(im)


def sac(lbl):
    return {int(v): (lbl == v) for v in np.unique(lbl) if LO <= v <= HI}


def main():
    frac = ["006", "063", "084", "198", "075", "108", "182", "097", "058"]  # fold_3 held-out (경합 회피로 한정)
    n_seam = 0
    has_anchor = 0          # max contrast >= 80 (어딘가 또렷)
    vis_fracs = []          # 보이는(>=40HU) 면 비율
    max_contrasts = []
    axial_parallel = 0      # 면 법선이 z축과 <30도 (axial 평면에 평행) = axial 불리
    normals_ang = []
    for c in frac:
        cid = f"PENGWIN_{c}"
        pp = Path(OOF) / f"{cid}.nii.gz"; gp = Path(RAW) / c / "label.mha"; ip = Path(RAW) / c / "image.mha"
        if not (pp.exists() and gp.exists() and ip.exists()):
            continue
        pim, sem = lps(pp); sp = pim.GetSpacing(); vv = float(np.prod(sp)); spz = (sp[2], sp[1], sp[0])
        _, gt = lps(gp); _, ct = lps(ip); ct = ct.astype(np.float32)
        inst = assemble_pelvic_instances_mode(sem, vv, mode="legacy", spacing_zyx=spz, min_volume_mm3=500)
        inst = remove_small_components(inst.astype(np.uint16), spz, 1000.0)
        gi = sac(gt); pi = sac(inst)
        if len(gi) < 2:
            continue
        for pid, pmask in pi.items():
            cov = [g for g, gm in gi.items() if (pmask & gm).sum() > 0.2 * gm.sum()]
            for i in range(len(cov)):
                for j in range(i + 1, len(cov)):
                    A, B = gi[cov[i]], gi[cov[j]]
                    seam = (binary_dilation(A, ST, 1) & B) | (binary_dilation(B, ST, 1) & A)
                    if seam.sum() < 5:
                        continue
                    n_seam += 1
                    bone = ct[(A | B)]; bone_med = float(np.median(bone[bone >= 0])) if (bone >= 0).any() else 200.0
                    sc = bone_med - ct[seam]      # voxel별 대조
                    vis_fracs.append(float((sc >= 40).mean()))
                    max_contrasts.append(float(np.percentile(sc, 90)))  # 90퍼센타일=anchor 후보
                    if np.percentile(sc, 90) >= 80:
                        has_anchor += 1
                    # 면 법선: seam voxel 좌표 PCA 최소분산축
                    pts = np.argwhere(seam).astype(np.float64) * np.array(spz)  # mm
                    if len(pts) >= 5:
                        pts -= pts.mean(0)
                        _, _, vt = np.linalg.svd(pts, full_matrices=False)
                        normal = vt[-1]  # 최소분산 = 법선
                        ang = np.degrees(np.arccos(min(1, abs(normal[0]))))  # z성분 → z축과의 각
                        normals_ang.append(ang)
                        if ang < 30:  # 법선이 z에 가까움 = 면이 axial 평면에 평행
                            axial_parallel += 1

    print(f"=== 놓친 seam {n_seam}개 — 부분 가시성/anchor + 방향 ===")
    if n_seam:
        print(f"(a) anchor 있음(90퍼센타일 대조>=80HU, 어딘가 또렷): {has_anchor}/{n_seam} = {100*has_anchor/n_seam:.0f}%")
        print(f"    보이는(>=40HU) 면 비율 중앙값: {100*np.median(vis_fracs):.0f}%  (0%=완전 안보임, 100%=다 보임)")
        print(f"    seam 90퍼센타일 대조 중앙값: {np.median(max_contrasts):.0f}HU  (이전 '중앙값 22'와 비교 — 면의 또렷한 부분)")
        va = np.array(vis_fracs)
        print(f"    부분가시(보이는비율 10~90%)={100*((va>0.1)&(va<0.9)).mean():.0f}%  완전안보임(<10%)={100*(va<=0.1).mean():.0f}%  거의다보임(>90%)={100*(va>=0.9).mean():.0f}%")
    if normals_ang:
        print(f"(b) 면 법선 z축과의 각 중앙값: {np.median(normals_ang):.0f}도  | axial-평행(법선~z, <30도)={100*axial_parallel/len(normals_ang):.0f}% (axial서 불리=직교뷰 유리)")
    # 판정
    if n_seam:
        anchorable = has_anchor / n_seam
        print(f"\n판정: anchor 보유 {100*anchorable:.0f}% → "
              f"{'면-완성/다중뷰 레버 유효 (부분 가시 seam을 이을 수 있음)' if anchorable >= 0.25 else '대부분 완전 안보임 → 면-완성도 한계'}")
    print("SEAM_ANCHOR_DONE")


if __name__ == "__main__":
    main()
