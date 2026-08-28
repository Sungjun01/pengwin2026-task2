# PENGWIN Evaluation Package

Implementation of PENGWIN 2024 evaluation metrics (paper §3.3) and multi-method
ranking scheme (§3.3.2). Used to compare baseline, ABBC, and our method on the
PENGWIN Task 1 validation set.

## Quick Start

### Single submission

```bash
python -m evaluation.pengwin_eval \
    -p /path/to/pred_dir \
    -g /path/to/gt_dir \
    -o metrics.json
```

Output (`metrics.json`):

```json
{
  "schema_version": "1.0",
  "per_case": {"case_001": {"IoU_F": 0.87, "HD95_F_mm": 2.3, ...}, ...},
  "aggregate": {"IoU_F_mean": 0.84, "IoU_F_median": 0.86, ...}
}
```

### Multi-method ranking

```bash
python -m evaluation.ranking \
    -s baseline:./pred_baseline \
    -s abbc:./pred_abbc \
    -s ours:./pred_ours \
    -g /path/to/gt_dir \
    -o ranking.json \
    --times baseline:120 abbc:180 ours:90
```

Output:
```
Ranking:
  1. ours (final rank score: 1.50)
  2. abbc (final rank score: 1.83)
  3. baseline (final rank score: 2.67)
```

### Programmatic API

```python
from evaluation import evaluate_case, evaluate_submission, rank_submissions

# Single case
metrics = evaluate_case("pred.nii.gz", "gt.nii.gz")

# Full submission directory
result = evaluate_submission("./pred", "./gt", output_path="metrics.json")

# Multi-method comparison
ranking = rank_submissions(
    submissions={"baseline": "./pred_baseline", "ours": "./pred_ours"},
    gt_dir="./gt",
)
```

## Architecture

- `metrics.py`: IoU, HD95, ASSD primitives (MONAI under the hood)
- `matching.py`: Hungarian (default) + greedy fragment matching
- `label_schema.py`: PENGWIN bone-range definitions (1-50 sacrum, etc.)
- `io_utils.py`: SimpleITK-backed mha/nii.gz loading
- `pengwin_eval.py`: F-level + A-level per-case eval; CLI
- `ranking.py`: Multi-method PENGWIN ranking scheme; CLI

## Metric Definitions

Per PENGWIN 2024 paper §3.3.1:

- **IoU** = |Sp ∩ Sg| / |Sp ∪ Sg|
- **HD95** = 95th percentile of bidirectional surface distance
- **ASSD** = Average Symmetric Surface Distance

**Fragment-level (F)**: per matched fragment instance, averaged over GT fragments.
Fragment matching: Hungarian (one-to-one with max total IoU).
Missing fragment penalty: HD95 = volume-equivalent bounding sphere diameter; ASSD = radius.

**Anatomical-level (A)**: merge all fragments per anatomy class (sacrum / leftHip / rightHip / femur),
compute IoU/HD95/ASSD per class, averaged over classes present in GT.

## Ranking Scheme (paper §3.3.2)

For each of the 6 metrics (IoU/HD95/ASSD × F/A), submissions are ranked
independently. The final rank is the **mean across all 6 metric ranks**.
Tie-breaker: lower container execution time wins.

## Verification

All primitive metrics unit-tested on synthetic cases with analytic ground truth.
HD95/ASSD computed via MONAI for cross-validation. See `tests/` for the full
test suite.

```bash
pytest evaluation/tests/ -v
```

## Spec

See `docs/specs/2026-05-14-evaluation-design.md`.
