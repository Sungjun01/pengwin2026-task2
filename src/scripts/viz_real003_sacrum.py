"""실제 public 003 사크럼 시각화 — CT + D016 / D019 / FS-DF 분할 비교.

사크럼(1-50) 조각이 가장 많이 보이는 axial 슬라이스에서, CT 위에 각 모델의
사크럼 instance 를 색으로 오버레이. FS-DF 가 어디를 더 가르는지 눈으로.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

CT = "/home/schoaq/taehee/bf2df8fe-0ac7-421b-a45f-aef664d00fb3 (1).mha"
D016 = "/tmp/real003/out_d016/images/peripelvic-fracture-ct-segmentation/real003.mha"
D019 = "/home/schoaq/taehee/1b84047a-450e-4402-b621-23d25bdd3253.mha"
FSDF = "/tmp/fsdf_real003_inst.nii.gz"


def lps_arr(p):
    return sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS"))


def sac_relabel(a):
    """사크럼(1-50)만 추려 1..N 으로 재라벨 (색 구분용)."""
    out = np.zeros_like(a)
    sac = [v for v in np.unique(a) if 1 <= v <= 50]
    for i, v in enumerate(sac, 1):
        out[a == v] = i
    return out, len(sac)


def main():
    ct = lps_arr(CT).astype(np.float32)
    d016, _ = None, None
    maps = {}
    for name, p in [("D016 (v18, LB 0.952)", D016), ("D019 (v22, LB 0.841)", D019), ("FS-DF (D020)", FSDF)]:
        try:
            maps[name] = lps_arr(p)
        except Exception as e:
            print("skip", name, e)

    # 사크럼이 가장 많이 보이는 z 슬라이스 (FS-DF 기준 — 조각 최다)
    ref = maps.get("FS-DF (D020)", next(iter(maps.values())))
    sac_ref = np.where((ref >= 1) & (ref <= 50), ref, 0)
    zc = [(len(np.unique(sac_ref[z])[np.unique(sac_ref[z]) > 0]), z) for z in range(sac_ref.shape[0])]
    z = max(zc)[1]
    print(f"사크럼 최다 슬라이스 z={z} (조각 {max(zc)[0]}개 보임)")

    n = len(maps)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5.5))
    # (0) CT 단독
    axes[0].imshow(ct[z], cmap="gray", vmin=-200, vmax=900)
    axes[0].set_title("(0) CT only\n(real public 003)")
    axes[0].axis("off")
    # 각 모델
    for i, (name, m) in enumerate(maps.items(), 1):
        sac, ns = sac_relabel(np.where((m >= 1) & (m <= 50), m, 0)[z][None])
        # z 슬라이스만 재라벨
        msac = np.where((m[z] >= 1) & (m[z] <= 50), m[z], 0)
        rl, ns = sac_relabel(msac)
        axes[i].imshow(ct[z], cmap="gray", vmin=-200, vmax=900)
        mask = np.ma.masked_where(rl == 0, rl)
        axes[i].imshow(mask, cmap="tab10", alpha=0.7, vmin=1, vmax=10)
        # 전체 볼륨 기준 사크럼 조각 수
        nfull = len([v for v in np.unique(m) if 1 <= v <= 50])
        axes[i].set_title(f"({i}) {name}\nsacrum {nfull} fragments")
        axes[i].axis("off")

    plt.suptitle("Real public case 003 — SACRUM fragmentation (the battleground)", fontsize=14)
    plt.tight_layout()
    out = "figs/real003_sacrum.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
