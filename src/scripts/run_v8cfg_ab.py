"""D017-sacrum A/B: v15 pelvic(ResEnc) + D017 ResEnc sacrum 으로 3 케이스 처리 후 채점."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, "/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee")
os.environ["PIVOT_PELVIC_DATASET"] = "Dataset015_PENGWIN_BorderCore_OrientFixed"
os.environ["PIVOT_PELVIC_TRAINER_PLANS"] = "nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres"
import SimpleITK as sitk, numpy as np
from scripts.inference_v7_d015 import run_pipeline_v7
from evaluation.pengwin_eval import evaluate_case

RES = Path("/home/schoaq/taehee/nnUNet_data/results")
OUT = Path("/tmp/v8cfg/out"); OUT.mkdir(parents=True, exist_ok=True)
PARTS = {'sac': (1, 50), 'LH': (51, 100), 'RH': (101, 150), 'fem': (151, 200)}

for cid in ["003", "017", "025"]:
    inp = Path(f"/tmp/v15_verify/in/PENGWIN_{cid}.mha")
    outp = OUT / f"PENGWIN_{cid}.mha"
    print(f"=== v8cfg(D015default+D006) pipeline PENGWIN_{cid} (exists={inp.exists()}) ===", flush=True)
    run_pipeline_v7(input_ct=inp, output_path=outp, nnunet_results=RES,
                    tile_step_size=1.0, use_pivot_form=False, use_d017_sacrum=False,
                    form_learn_mode=None, form_tau_intact=0.08, min_volume_mm3=100.0)

def lps(p):
    img = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False).name
    sitk.WriteImage(img, f); return f, sitk.GetArrayFromImage(img).astype(int)
def cnt(a):
    u = set(int(v) for v in np.unique(a)) - {0}
    return {p: sum(1 for v in u if lo <= v <= hi) for p, (lo, hi) in PARTS.items()}

print("\n===== v8cfg SCORES =====", flush=True)
ld, rq = [], []
for cid in ["003", "017", "025"]:
    pf, pa = lps(OUT / f"PENGWIN_{cid}.mha")
    gf, ga = lps(f"/home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee/predictions/gt_instance/PENGWIN_{cid}.nii.gz")
    r = evaluate_case(pf, gf); pc, gc = cnt(pa), cnt(ga)
    parts = " ".join(f"{p}{pc[p]}/{gc[p]}" for p in PARTS if gc[p] or pc[p])
    print(f"PENGWIN_{cid}  LDSC={r['LDSC_F']:.3f} RQ={r['RQ_F']:.3f} HD95={r['HD95_F_mm']:.2f}  {parts}", flush=True)
    ld.append(r['LDSC_F']); rq.append(r['RQ_F'])
print(f"MEAN(v8cfg): LDSC={np.mean(ld):.3f} RQ={np.mean(rq):.3f}", flush=True)
print("V8CFG_AB_DONE", flush=True)
