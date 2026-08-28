#!/usr/bin/env bash
# Task A (Dataset007 vanilla cascade) best checkpoint 로 4 sample case inference + viz.
# GPU 5 (Task D 학습 중) 의 메모리 여유 (49 GB GPU) 안에서 병행 실행.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/nnunet_env.sh"

GPU="${GPU:-5}"
CASES=(PENGWIN_001 PENGWIN_050 PENGWIN_100 PENGWIN_200)

OUT_ROOT="$PROJ/viz/exp5_d007_predictions"
PRED="$OUT_ROOT/d007_vanilla_pred"
VIZ="$OUT_ROOT/d007_vanilla_viz"
mkdir -p "$PRED" "$VIZ"

# nnUNet 의 predict 는 input dir 의 *_0000.nii.gz, *_0001.nii.gz 둘 다 있어야 (2-channel D007)
TMP_IN="/tmp/viz_d007_input"
rm -rf "$TMP_IN" && mkdir -p "$TMP_IN"
SRC_IMG="$nnUNet_raw/Dataset007_PENGWIN_PelvicCascade/imagesTr"

for case in "${CASES[@]}"; do
    for ch in 0000 0001; do
        src="$SRC_IMG/${case}_${ch}.nii.gz"
        if [[ -f "$src" ]]; then
            cp "$src" "$TMP_IN/"
        else
            echo "[warn] $src 없음 (case $case channel $ch), skip"
        fi
    done
done

n_in=$(ls "$TMP_IN" | wc -l)
echo "[setup] GPU=$GPU, input files=$n_in (2 ch × cases), out=$OUT_ROOT"

echo "=== D007 vanilla cascade inference (Task A best EMA) ==="
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict \
    -i "$TMP_IN" -o "$PRED" \
    -d 7 -c 3d_fullres -tr nnUNetTrainer -f all \
    -chk checkpoint_best.pth

echo "=== D007 viz (4-class anatomy: bg/sacrum/leftHip/rightHip) ==="
python3 "$PROJ/scripts/visualize_predictions.py" \
    --ct_dir   "$nnUNet_raw/Dataset003_PENGWIN_PelvicAnatomy/imagesTr" \
    --gt_dir   "$nnUNet_raw/Dataset003_PENGWIN_PelvicAnatomy/labelsTr" \
    --pred_dir "$PRED" \
    --out_dir  "$VIZ" \
    --cases "${CASES[@]}"

rm -rf "$TMP_IN"
echo "DONE — viz at $OUT_ROOT"
ls -la "$VIZ"
