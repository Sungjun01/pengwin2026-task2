import numpy as np

from evaluation.tests.fixtures.synthetic_cases import (
    two_overlapping_cubes,
    two_disjoint_cubes_distance_mm,
    n_disjoint_fragments,
    case_with_missing_fragment,
    perfect_prediction,
)


def test_two_overlapping_cubes_iou_target():
    pred, gt, _ = two_overlapping_cubes(overlap_fraction=0.5)
    inter = ((pred > 0) & (gt > 0)).sum()
    union = ((pred > 0) | (gt > 0)).sum()
    assert abs(inter / union - 0.5) < 0.05  # discretization tolerance


def test_two_disjoint_cubes_distance():
    pred, gt, _ = two_disjoint_cubes_distance_mm(distance_mm=5.0)
    assert (pred > 0).sum() == 1 and (gt > 0).sum() == 1
    assert not np.array_equal(pred > 0, gt > 0)


def test_n_disjoint_fragments_count_and_offset():
    lbl, _ = n_disjoint_fragments(n=4, anatomy_offset=50)
    ids = sorted(set(np.unique(lbl)) - {0})
    assert ids == [51, 52, 53, 54]


def test_missing_fragment_counts():
    pred, gt, _ = case_with_missing_fragment(n_gt=3, n_pred=2)
    gt_ids = set(np.unique(gt)) - {0}
    pred_ids = set(np.unique(pred)) - {0}
    assert len(gt_ids) == 3 and len(pred_ids) == 2


def test_perfect_prediction_identical():
    pred, gt, _ = perfect_prediction(n=3)
    assert np.array_equal(pred, gt)
