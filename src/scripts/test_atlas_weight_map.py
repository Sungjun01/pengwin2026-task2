"""atlas-가중 W-map sanity 테스트 — 학습 전 축치환/orientation 버그 검출.

합성 target(중심에 sac_core blob)으로 _atlas_weight_map 을 만들어:
  1. W 범위 [base, base+γ] 정상?
  2. W 가 균일하지 않나(atlas 신호 반영)?
  3. **축치환 정확성**: W argmax 의 patch-축 offset → atlas-축으로 역매핑 → atlas argmax offset 과 일치?
  4. weighted_tversky 에 grad 흐르나(recall 방향)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gaussian_recall_loss import GaussianSkeletonRecallLoss, weighted_tversky

ATLAS = "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/assets/fracture_atlas.npz"
SPACING = [0.8125, 0.800048828125, 0.8125]  # patch축 [a0,a1,a2] mm (config 3d_fullres)
TF = [1, 0, 2]                                # plans transpose_forward
P = (112, 128, 160)                            # patch_size


def main():
    loss = GaussianSkeletonRecallLoss(mode="atlas", atlas_path=ATLAS, atlas_base=0.3,
                                      atlas_gamma=2.0, core_index=1, spacing=SPACING,
                                      transpose_forward=TF)
    d = np.load(ATLAS); atlas = d["atlas"]; R = float(d["R"]); BN = float(d["BIN"]); N = int(d["N"])
    print(f"atlas shape={atlas.shape} max={atlas.max():.3f} mean={atlas.mean():.3f}")

    # 합성 target: sac_core(=1) blob 중심
    tgt = torch.zeros((1, 1) + P, dtype=torch.long)
    c0, c1, c2 = P[0] // 2, P[1] // 2, P[2] // 2
    tgt[0, 0, c0 - 6:c0 + 6, c1 - 6:c1 + 6, c2 - 6:c2 + 6] = 1  # core blob
    W = loss._atlas_weight_map(tgt)[0, 0].numpy()
    print(f"\n[1] W 범위: min={W.min():.3f} max={W.max():.3f} (기대 base=0.3 ~ base+γ=2.3)")
    print(f"[2] W std={W.std():.3f} (>0.05 면 atlas 신호 반영, uniform 아님)")

    # [3] 축치환 정확성: W argmax 의 patch offset → atlas axis 로 역매핑 → atlas argmax 와 비교
    wmax = np.unravel_index(np.argmax(W), W.shape)           # patch (p0,p1,p2)
    off_patch_mm = [(wmax[i] - [c0, c1, c2][i]) * SPACING[i] for i in range(3)]
    # patch axis ax 는 atlas axis tf[ax] 를 담음 → atlas offset[tf[ax]] = off_patch_mm[ax]
    atlas_off = [0, 0, 0]
    for ax in range(3):
        atlas_off[TF[ax]] = off_patch_mm[ax]
    amax = np.unravel_index(np.argmax(atlas), atlas.shape)   # atlas (iz,iy,ix)
    atlas_off_true = [(amax[k] * BN - R + BN / 2) for k in range(3)]  # bin 중심 mm (z,y,x)
    print(f"\n[3] 축치환 검증:")
    print(f"    W argmax patch voxel={wmax}, patch offset(mm)={[round(o,1) for o in off_patch_mm]}")
    print(f"    → atlas축 역매핑 offset(z,y,x mm)={[round(o,1) for o in atlas_off]}")
    print(f"    atlas 자체 argmax offset(z,y,x mm)={[round(o,1) for o in atlas_off_true]}")
    err = np.sqrt(sum((atlas_off[k] - atlas_off_true[k]) ** 2 for k in range(3)))
    print(f"    오차={err:.1f}mm  → {'PASS(축치환 정확)' if err < 10 else 'FAIL(축 어긋남!)'}")

    # [4] grad 흐름 (recall 방향)
    p = torch.zeros((1, 1) + P, requires_grad=True)
    g = torch.zeros((1, 1) + P); g[0, 0, c0 - 3:c0 + 3, c1 - 3:c1 + 3, c2 - 3:c2 + 3] = 1.0  # GT seam (FN)
    w = torch.from_numpy(W).view((1, 1) + P)
    l = weighted_tversky(torch.sigmoid(p), g, w, 0.3, 0.7)
    l.backward()
    gm = p.grad[0, 0, c0 - 3:c0 + 3, c1 - 3:c1 + 3, c2 - 3:c2 + 3].mean().item()
    print(f"\n[4] weighted_tversky grad on FN region mean={gm:+.5f} (음수=border_prob↑=recall) "
          f"→ {'PASS' if gm < 0 else 'FAIL'}")
    print("ATLAS_WMAP_TEST_DONE")


if __name__ == "__main__":
    main()
