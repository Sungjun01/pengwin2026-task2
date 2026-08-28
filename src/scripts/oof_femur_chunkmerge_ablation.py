"""femur chunk-merge ablation — PIVOT-Assembly의 merge_femur_chunks가 *골절 femur* 조각을
붕괴시켜 인스턴스 점수를 깎는지 held-out OOF로 검증.

같은 34 held-out femur 케이스에서:
  A = watershed + sliver-merge          (chunk-merge OFF)
  B = watershed + sliver-merge + chunk-merge   (= V18 production)
per-case RQ_F(ins_f1)/HD95_F + 인스턴스 개수 비교. B<A 면 chunk-merge가 해.

용법: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 -m scripts.oof_femur_chunkmerge_ablation
"""
from __future__ import annotations
import sys, os, json, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scripts.inference_v7_d015 import predict_softmax, per_anatomy_watershed_instance
from scripts.pivot_assembly import (
    AssemblyConfig, merge_sliver_fragments, merge_femur_chunks_by_spatial_groups)
from evaluation.pengwin_eval import evaluate_case

FEMUR = (151, 200)
MIN_VOL = 500.0
RES = Path("/home/schoaq/taehee/nnUNet_data/results/Dataset004_PENGWIN_FemurAnatomy")
RAW = Path("/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/raw_data/PENGWIN_train")
SPLITS = Path("/home/schoaq/taehee/nnUNet_data/preprocessed/Dataset004_PENGWIN_FemurAnatomy/splits_final.json")
CKPT_DIR = os.environ.get("PENGWIN_OOF_CKPT",
    str(RES / "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"))
ACFG = AssemblyConfig(max_sliver_volume_mm3=2000.0, min_volume_ratio=5.0,
                      dilation_iters=1, max_centroid_distance_mm=30.0)


def init_pred(d, fold="0"):
    p = nnUNetPredictor(tile_step_size=1.0, use_gaussian=False, use_mirroring=False,
                        perform_everything_on_device=True, device=torch.device("cuda", 0),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(d, use_folds=(fold,), checkpoint_name="checkpoint_best.pth")
    return p


def wlps(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint16)); img.CopyInformation(ref)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f


def n_inst(a):
    return len([v for v in np.unique(a) if 151 <= v <= 200])


def main():
    cids = [v.split("_")[-1] for v in json.load(open(SPLITS))[0]["val"]]
    gtN = {}
    pred = init_pred(CKPT_DIR)
    tmp = Path(tempfile.mkdtemp(prefix="chunkabl_"))
    print(f"[ablation] ckpt={Path(CKPT_DIR).name}  n={len(cids)}", flush=True)
    print(f"{'case':6s} {'gtN':>3s} {'nA':>3s} {'nB':>3s} {'RQ_A':>6s} {'RQ_B':>6s} {'ΔRQ':>6s} "
          f"{'HD95_A':>7s} {'HD95_B':>7s}  flag", flush=True)
    rows = []
    for cid in cids:
        ip = RAW / cid / "image.mha"
        if not ip.exists():
            continue
        ct = sitk.DICOMOrient(sitk.ReadImage(str(ip)), "LPS")
        gt = sitk.DICOMOrient(sitk.ReadImage(str(RAW / cid / "label.mha")), "LPS")
        sx, sy, sz = ct.GetSpacing(); vv = float(sx*sy*sz); spz = (sz, sy, sx)
        gt_arr = sitk.GetArrayFromImage(gt).astype(int)
        gN = len([v for v in np.unique(gt_arr) if 151 <= v <= 200])
        cp = tmp/"ct_0000.nii.gz"; sitk.WriteImage(ct, str(cp))
        probs = predict_softmax(pred, cp, tmp/"o")
        fem = (probs[1].astype(np.float32) > 0.5).astype(np.uint8)
        ws = per_anatomy_watershed_instance(fem, 1, FEMUR, vv, MIN_VOL)
        # A = watershed + sliver
        A = merge_sliver_fragments(ws, vv, ACFG, spz)
        # B = A + chunk-merge (V18)
        B = merge_femur_chunks_by_spatial_groups(A, vv, spz)
        gtf = wlps(np.where((gt_arr>=151)&(gt_arr<=200), gt_arr, 0), gt)
        rA = evaluate_case(wlps(A, ct), gtf); rB = evaluate_case(wlps(B, ct), gtf)
        dRQ = rB['RQ_F'] - rA['RQ_F']
        flag = "CHUNK-FIRED" if n_inst(A) != n_inst(B) else ""
        if dRQ < -0.01: flag += " ⬇HURT"
        elif dRQ > 0.01: flag += " ⬆HELP"
        print(f"{cid:6s} {gN:>3d} {n_inst(A):>3d} {n_inst(B):>3d} {rA['RQ_F']:6.3f} {rB['RQ_F']:6.3f} "
              f"{dRQ:+6.3f} {rA['HD95_F_mm']:7.2f} {rB['HD95_F_mm']:7.2f}  {flag}", flush=True)
        rows.append((cid, gN, rA, rB, n_inst(A), n_inst(B)))
    # 집계
    import numpy as _np
    rqA = _np.mean([r[2]['RQ_F'] for r in rows]); rqB = _np.mean([r[3]['RQ_F'] for r in rows])
    hdA = _np.mean([r[2]['HD95_F_mm'] for r in rows]); hdB = _np.mean([r[3]['HD95_F_mm'] for r in rows])
    fired = [r for r in rows if r[4] != r[5]]
    hurt = [r for r in rows if r[3]['RQ_F'] - r[2]['RQ_F'] < -0.01]
    print(f"\n=== 집계 (n={len(rows)}) ===")
    print(f"  RQ_F:   A(no-chunk)={rqA:.3f}   B(+chunk)={rqB:.3f}   Δ={rqB-rqA:+.3f}")
    print(f"  HD95_F: A={hdA:.2f}   B={hdB:.2f}   Δ={hdB-hdA:+.2f}")
    print(f"  chunk-merge 발동: {len(fired)}/{len(rows)} 케이스, 그 중 RQ 악화: {len(hurt)}")
    print("CHUNKABL_DONE")


if __name__ == "__main__":
    main()
