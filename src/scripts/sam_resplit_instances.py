"""neck-gated negative-prompt SAM re-split of cached fold_0 instance maps (워크플로우 계획 구현).
캐시 인스턴스맵(이미 watershed 디코드됨)에서 merge-의심(2개 DT-peak + neck 없음) 인스턴스를
base-SAM(sam_vit_b) neg-prompt(pos@peakA + neg@peakB)로 재분리.
안전장치: SAM 실패/sub-fragment<min_vox → 원본 그대로(조각 절대 삭제 안 함, recall 가드).
CT를 인스턴스맵 grid로 resample → LPS 정합 자동(인덱싱 어긋남 방지). 출력 = 같은 geometry, .nii.gz(기존 러너 호환).
용법: CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=N python3 scripts/sam_resplit_instances.py \
        --inst-dir proxy_out/aseg_fold0_decoded_instances --ct-root raw_data/PENGWIN_train \
        --out-dir proxy_out/aseg_fold0_sam_resplit_t065 [--neck-thresh 0.65]
"""
import os, argparse, time, glob
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")
from segment_anything import sam_model_registry, SamPredictor

BONE_BLOCKS = [(1, 50), (51, 100), (101, 150)]


def bone_block(lab):
    for lo, hi in BONE_BLOCKS:
        if lo <= lab <= hi:
            return lo, hi
    return None


def neck_ratio_2d(m2d):
    """central slice neck 강도: watershed 경계 max-DT / blob max-DT. ↑이면 목 없음."""
    dt = ndi.distance_transform_edt(m2d)
    pk = peak_local_max(dt, min_distance=6, num_peaks=2, labels=m2d.astype(int))
    if len(pk) < 2 or dt.max() == 0:
        return 0.0
    mk = np.zeros(m2d.shape, int)
    for i, p in enumerate(pk):
        mk[p[0], p[1]] = i + 1
    ws = watershed(-dt, mk, mask=m2d)
    w1, w2 = (ws == 1), (ws == 2)
    bnd = (ndi.binary_dilation(w1) & w2) | (ndi.binary_dilation(w2) & w1)
    return dt[bnd].max() / dt.max() if bnd.any() else 1.0


def try_resplit(inst_mask, ct, predictor, neck_thresh, min_sub_vox=200):
    """2개 3D-peak + neck 없음이면 SAM neg-prompt로 재분리. (subA, subB) 또는 None."""
    dt3 = ndi.distance_transform_edt(inst_mask)
    peaks = peak_local_max(dt3, min_distance=20, num_peaks=2, labels=inst_mask.astype(int))
    if len(peaks) < 2:
        return None
    pA, pB = peaks[0], peaks[1]                                   # (z,y,x)
    zr = np.where(inst_mask.any(axis=(1, 2)))[0]
    zc = int(zr[np.argmax([dt3[z].max() for z in zr])])
    if neck_ratio_2d(inst_mask[zc]) < neck_thresh:               # 목 있음 → watershed 충분
        return None
    # 기본 3D 배정 = 가까운 peak (기하)
    zz, yy, xx = np.where(inst_mask)
    dA = (zz - pA[0])**2 + (yy - pA[1])**2 + (xx - pA[2])**2
    dB = (zz - pB[0])**2 + (yy - pB[1])**2 + (xx - pB[2])**2
    asn = np.full(inst_mask.shape, -1, np.int8)
    asn[zz, yy, xx] = np.where(dA <= dB, 0, 1)
    # 중심 슬라이스(상위 DT 15장)에서 SAM 으로 override
    for z in sorted(zr, key=lambda z: -dt3[z].max())[:15]:
        m = inst_mask[z]
        if not m.any():
            continue
        img = np.clip(ct[z], -200, 1000); img = ((img + 200) / 1200 * 255).astype(np.uint8)
        rgb = np.stack([img] * 3, -1)
        try:
            predictor.set_image(rgb)
            mA, _, _ = predictor.predict(point_coords=np.array([[pA[2], pA[1]], [pB[2], pB[1]]]),
                                         point_labels=np.array([1, 0]), multimask_output=False)
            mB, _, _ = predictor.predict(point_coords=np.array([[pB[2], pB[1]], [pA[2], pA[1]]]),
                                         point_labels=np.array([1, 0]), multimask_output=False)
            rA = mA[0] & m; rB = mB[0] & m
        except Exception:
            continue
        if rA.sum() < 10 or rB.sum() < 10:
            continue
        sl = asn[z]
        sl[rA & ~rB] = 0; sl[rB & ~rA] = 1
        ov = rA & rB
        if ov.any():
            y2, x2 = np.where(ov)
            d1 = (y2 - pA[1])**2 + (x2 - pA[2])**2; d2 = (y2 - pB[1])**2 + (x2 - pB[2])**2
            sl[y2[d1 <= d2], x2[d1 <= d2]] = 0; sl[y2[d1 > d2], x2[d1 > d2]] = 1
        asn[z] = sl
    subA, subB = (asn == 0), (asn == 1)
    if subA.sum() < min_sub_vox or subB.sum() < min_sub_vox:
        return None
    return subA, subB


def resplit_case(inst_path, ct_path, out_path, predictor, neck_thresh):
    inst_img = sitk.ReadImage(inst_path)
    inst = sitk.GetArrayFromImage(inst_img)
    ct_img = sitk.ReadImage(ct_path)
    ct_img = sitk.Resample(ct_img, inst_img, sitk.Transform(), sitk.sitkLinear, -1000.0, ct_img.GetPixelID())
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    out = inst.copy(); n_app = 0; t0 = time.time()
    for L in [int(l) for l in np.unique(inst) if 1 <= l <= 150]:
        bb = bone_block(L)
        if bb is None:
            continue
        m = (inst == L)
        if m.sum() < 400:
            continue
        res = try_resplit(m, ct, predictor, neck_thresh)
        if res is None:
            continue
        _, subB = res
        used = set(int(x) for x in np.unique(out) if bb[0] <= x <= bb[1])
        free = next((i for i in range(bb[0], bb[1] + 1) if i not in used), None)
        if free is None:
            continue
        out[subB] = free; n_app += 1                              # subA 는 L 유지
    o = sitk.GetImageFromArray(out.astype(np.uint16)); o.CopyInformation(inst_img)
    sitk.WriteImage(o, out_path, useCompression=True)
    return n_app, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inst-dir', required=True); ap.add_argument('--ct-root', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--neck-thresh', type=float, default=0.65)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sam = sam_model_registry["vit_b"](checkpoint="/home/schoaq/taehee/sam_ckpt/sam_vit_b.pth").to("cuda").eval()
    predictor = SamPredictor(sam)
    files = sorted(glob.glob(f"{a.inst_dir}/PENGWIN_*.mha"))
    print(f"케이스 {len(files)}개, neck_thresh={a.neck_thresh}", flush=True)
    tot = 0
    for f in files:
        case = os.path.basename(f).replace("PENGWIN_", "").replace(".mha", "")
        ct = f"{a.ct_root}/{case}/image.mha"
        if not os.path.exists(ct):
            print(f"  {case}: CT 없음 skip", flush=True); continue
        n, sec = resplit_case(f, ct, f"{a.out_dir}/PENGWIN_{case}.nii.gz", predictor, a.neck_thresh)
        tot += n
        print(f"  {case}: 재분리 {n}개 ({sec:.0f}s)", flush=True)
    print(f"SAM_RESPLIT_DONE 총 재분리 {tot}개", flush=True)


if __name__ == "__main__":
    main()
