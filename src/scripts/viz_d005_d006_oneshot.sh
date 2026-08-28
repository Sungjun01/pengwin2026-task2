#!/usr/bin/env bash
# D005 (3-class sacrum + hipbone L+R merged) + D006 (2-class sacrum binary) best checkpoint
# 으로 5 sample case 추론 + PNG viz 생성.
# 학습 완료 후 1회 batch viz 용. D005/D006 모두 학습 완료된 상태에서 실행.
#
# Usage: GPU=6 bash scripts/viz_d005_d006_oneshot.sh
# 환경변수 GPU 미지정 시 6 사용 (D005 학습 끝나면 자유 GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/nnunet_env.sh"

GPU="${GPU:-6}"
CASES=(PENGWIN_001 PENGWIN_050 PENGWIN_100 PENGWIN_150 PENGWIN_200)

OUT_ROOT="$PROJ/viz/exp3_d005_d006"
D005_PRED="$OUT_ROOT/d005_pred"
D006_PRED="$OUT_ROOT/d006_pred"
D005_VIZ="$OUT_ROOT/d005_viz"
D006_VIZ="$OUT_ROOT/d006_viz"
mkdir -p "$D005_PRED" "$D006_PRED" "$D005_VIZ" "$D006_VIZ"

TMP_IN="/tmp/viz_oneshot_input"
rm -rf "$TMP_IN" && mkdir -p "$TMP_IN"

D003_IMG="$nnUNet_raw/Dataset003_PENGWIN_PelvicAnatomy/imagesTr"
D003_LBL="$nnUNet_raw/Dataset003_PENGWIN_PelvicAnatomy/labelsTr"
D006_LBL="$nnUNet_raw/Dataset006_PENGWIN_SacrumOnly/labelsTr"

for case in "${CASES[@]}"; do
    src="$D003_IMG/${case}_0000.nii.gz"
    if [[ -f "$src" ]]; then
        cp "$src" "$TMP_IN/"
    else
        echo "[warn] $src 없음, skip"
    fi
done

n_in=$(ls "$TMP_IN" | wc -l)
echo "[setup] GPU=$GPU, input cases=$n_in, out=$OUT_ROOT"

echo "=== D005 (3-class) inference ==="
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict \
    -i "$TMP_IN" -o "$D005_PRED" \
    -d 005 -c 3d_fullres -tr nnUNetTrainer -f all \
    -chk checkpoint_best.pth

echo "=== D006 (2-class) inference ==="
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict \
    -i "$TMP_IN" -o "$D006_PRED" \
    -d 006 -c 3d_fullres -tr nnUNetTrainer -f all \
    -chk checkpoint_best.pth

echo "=== D005 viz (GT: D003 4-class) ==="
python3 "$PROJ/scripts/visualize_predictions.py" \
    --ct_dir   "$D003_IMG" \
    --gt_dir   "$D003_LBL" \
    --pred_dir "$D005_PRED" \
    --out_dir  "$D005_VIZ" \
    --cases "${CASES[@]}"

echo "=== D006 viz (GT: D006 sacrum binary) ==="
python3 "$PROJ/scripts/visualize_predictions.py" \
    --ct_dir   "$D003_IMG" \
    --gt_dir   "$D006_LBL" \
    --pred_dir "$D006_PRED" \
    --out_dir  "$D006_VIZ" \
    --cases "${CASES[@]}"

rm -rf "$TMP_IN"
echo "DONE — viz at $OUT_ROOT"
ls "$D005_VIZ" "$D006_VIZ"
