#!/usr/bin/env bash
# 5-way ablation 의 170 case full 8-metric evaluation.
# 각 model 의 best EMA 로 batch inference → Task I postprocess (1-200 instance encoding)
# → evaluation/pengwin_eval.py 의 8-metric (IoU/HD95/ASSD × A + IoU/HD95/ASSD/LDSC/RQ × F)
#
# 용법.
#   GPU=5 bash scripts/eval_5way_8metric.sh d007    # Task A (vanilla)
#   GPU=5 bash scripts/eval_5way_8metric.sh d009    # Task D (Oracle)
#   GPU=5 bash scripts/eval_5way_8metric.sh d008    # Task F (Spatial Moments)
#   GPU=6 bash scripts/eval_5way_8metric.sh d010    # V1 Predicted
#   GPU=6 bash scripts/eval_5way_8metric.sh d011    # V1 Oracle

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/nnunet_env.sh"

if [[ -z "${1:-}" ]]; then
    echo "사용법: bash scripts/eval_5way_8metric.sh <d007|d008|d009|d010|d011>" >&2
    exit 1
fi
MODEL="$1"
GPU="${GPU:-5}"

case "$MODEL" in
    d007) DS_ID=7;  DS_NAME="Dataset007_PENGWIN_PelvicCascade"; SHORT="vanilla";;
    d008) DS_ID=8;  DS_NAME="Dataset008_PENGWIN_SpatialMoments"; SHORT="spatial_moments";;
    d009) DS_ID=9;  DS_NAME="Dataset009_PENGWIN_OracleSacrum"; SHORT="oracle";;
    d010) DS_ID=10; DS_NAME="Dataset010_PENGWIN_CentroidOnly_Predicted"; SHORT="v1_pred";;
    d011) DS_ID=11; DS_NAME="Dataset011_PENGWIN_CentroidOnly_Oracle"; SHORT="v1_oracle";;
    *) echo "unknown model: $MODEL"; exit 1;;
esac

INPUT_DIR="$nnUNet_raw/$DS_NAME/imagesTr"
PRED_RAW_DIR="$PROJ/predictions/${SHORT}_raw"
PRED_INSTANCE_DIR="$PROJ/predictions/${SHORT}_instance"
EVAL_JSON="$PROJ/predictions/${SHORT}_eval.json"
mkdir -p "$PRED_RAW_DIR" "$PRED_INSTANCE_DIR" "$PROJ/predictions"

echo "============================================================"
echo "[eval] model = $MODEL ($SHORT), DS_ID = $DS_ID"
echo "       input  = $INPUT_DIR"
echo "       pred   = $PRED_RAW_DIR"
echo "       inst   = $PRED_INSTANCE_DIR"
echo "       GPU = $GPU"
echo "============================================================"

# Stage 1 — batch inference (fast config: TTA off + step 1.0)
echo ""
echo "[Stage 1] nnUNetv2_predict batch inference (170 case)"
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict \
    -i "$INPUT_DIR" \
    -o "$PRED_RAW_DIR" \
    -d "$DS_ID" -c 3d_fullres -tr nnUNetTrainer -f all \
    -chk checkpoint_best.pth \
    --disable_tta \
    -step_size 1.0

# Stage 2 — Task I postprocess (4-class semantic → PENGWIN 1-200 instance encoding)
echo ""
echo "[Stage 2] postprocess_to_fragments → 1-200 instance encoding"
python3 "$SCRIPT_DIR/postprocess_to_fragments.py" \
    --pred_dir "$PRED_RAW_DIR" \
    --out_dir "$PRED_INSTANCE_DIR" \
    --min_volume_mm3 500.0

echo ""
echo "[done] prediction at $PRED_INSTANCE_DIR (8-metric eval 은 ranking script 에서 통합)"
