#!/usr/bin/env bash
# Gate a training snapshot: predict the 34 fold_0 val cases with <checkpoint>, then
# decode + score vs the champion baseline (proxy_out/run_fold0_gate.py).
# Usage: gate_snapshot.sh <trainer_name> <checkpoint_file.pth> <gpu> <tag>
set -uo pipefail
cd /home/schoaq/taehee/PENGWIN2026_Task1_AutoSeg_Baseline_taehee
source scripts/nnunet_env.sh 2>/dev/null
export PYTHONPATH="$PWD:$PWD/scripts/trainers:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nnUNet_n_proc_DA=4
TR=$1; CHK=$2; GPU=${3:-6}; TAG=${4:-snap}
DS=Dataset016_PENGWIN_BorderCore_MedialOrient
L=logs/gate_${TAG}.log; mkdir -p logs proxy_out; : > "$L"
TMPIN=proxy_out/gate_${TAG}_in; TMPOUT=proxy_out/gate_${TAG}_pred
rm -rf "$TMPIN" "$TMPOUT"; mkdir -p "$TMPIN"

echo "[gate] $(date +%H:%M:%S) tr=$TR chk=$CHK gpu=$GPU tag=$TAG" >> "$L"
# 1) link the 34 fold_0 val cases (5 channels each)
python3 - "$TMPIN" >> "$L" 2>&1 <<'PY'
import json, os, sys
tmpin = sys.argv[1]
pre = os.environ["nnUNet_preprocessed"] + "/Dataset016_PENGWIN_BorderCore_MedialOrient"
raw = os.environ["nnUNet_raw"] + "/Dataset016_PENGWIN_BorderCore_MedialOrient/imagesTr"
val = json.load(open(pre + "/splits_final.json"))[0]["val"]
n = 0
for c in val:
    for ch in range(5):
        f = f"{raw}/{c}_{ch:04d}.nii.gz"
        if os.path.exists(f):
            os.symlink(f, f"{tmpin}/{c}_{ch:04d}.nii.gz")
    n += 1
print(f"linked {n} val cases (5ch)")
PY

# 2) predict (no TTA — matches v18 inference)
echo "[gate] predict ..." >> "$L"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  nnUNetv2_predict -i "$TMPIN" -o "$TMPOUT" -d "$DS" -c 3d_fullres -f 0 \
  -tr "$TR" -chk "$CHK" --disable_tta -npp 2 -nps 2 >> "$L" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[gate] PREDICT FAILED rc=$rc" >> "$L"; exit $rc; fi

# 3) decode + score vs champion
echo "[gate] score ..." >> "$L"
python3 proxy_out/run_fold0_gate.py "$TMPOUT" "proxy_out/gate_${TAG}_score" >> "$L" 2>&1
echo "GATE_${TAG}_DONE rc=$? $(date +%H:%M:%S)" >> "$L"
