import numpy as np

from evaluation.matching import compute_pairwise_iou, match_fragments


def test_pairwise_iou_identity():
    lbl = np.zeros((10, 10, 10), dtype=np.int32)
    lbl[1:4, :, :] = 1
    lbl[5:8, :, :] = 2
    iou = compute_pairwise_iou(lbl, lbl, gt_ids=[1, 2], pred_ids=[1, 2])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(iou, expected, atol=1e-6)


def test_pairwise_iou_swapped_ids():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    gt[1:4, :, :] = 1
    gt[5:8, :, :] = 2
    pred = np.zeros_like(gt)
    pred[1:4, :, :] = 2
    pred[5:8, :, :] = 1
    iou = compute_pairwise_iou(gt, pred, gt_ids=[1, 2], pred_ids=[1, 2])
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(iou, expected, atol=1e-6)


def test_pairwise_iou_empty():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    pred = np.zeros_like(gt)
    iou = compute_pairwise_iou(gt, pred, gt_ids=[], pred_ids=[])
    assert iou.shape == (0, 0)


def test_match_fragments_hungarian_identity():
    lbl = np.zeros((10, 10, 10), dtype=np.int32)
    lbl[1:4, :, :] = 1
    lbl[5:8, :, :] = 2
    m = match_fragments(lbl, lbl, method="hungarian")
    assert m == {1: 1, 2: 2}


def test_match_fragments_hungarian_swapped():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    gt[1:4, :, :] = 1
    gt[5:8, :, :] = 2
    pred = np.zeros_like(gt)
    pred[1:4, :, :] = 2
    pred[5:8, :, :] = 1
    m = match_fragments(gt, pred, method="hungarian")
    assert m == {1: 2, 2: 1}


def test_match_fragments_more_pred_than_gt():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    gt[1:4, :, :] = 1
    pred = gt.copy()
    pred[7:9, :, :] = 9
    m = match_fragments(gt, pred, method="hungarian")
    assert m == {1: 1}


def test_match_fragments_more_gt_than_pred():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    gt[1:4, :, :] = 1
    gt[5:8, :, :] = 2
    pred = np.zeros_like(gt)
    pred[1:4, :, :] = 1
    m = match_fragments(gt, pred, method="hungarian")
    assert m == {1: 1, 2: None}


def test_match_fragments_hungarian_beats_greedy_on_traps():
    """Construct IoU matrix where greedy is strictly worse than Hungarian.

    iou = [[0.9, 0.85],
           [0.88, 0.10]]
    Greedy from GT0: pick pred0 (0.9), GT1 stuck with pred1 (0.10). Total 1.00.
    Hungarian: GT0->pred1 (0.85), GT1->pred0 (0.88). Total 1.73.
    """
    from evaluation.matching import _greedy_from_iou, _hungarian_from_iou
    iou = np.array([[0.9, 0.85], [0.88, 0.10]])
    greedy = _greedy_from_iou(iou, [1, 2], [10, 20])
    hungarian = _hungarian_from_iou(iou, [1, 2], [10, 20])
    greedy_score = (
        iou[0, 0 if greedy[1] == 10 else 1]
        + iou[1, 0 if greedy[2] == 10 else 1]
    )
    hung_score = (
        iou[0, 0 if hungarian[1] == 10 else 1]
        + iou[1, 0 if hungarian[2] == 10 else 1]
    )
    assert hung_score > greedy_score


def test_match_fragments_zero_iou_returns_none():
    gt = np.zeros((10, 10, 10), dtype=np.int32)
    gt[1:3, :, :] = 1
    pred = np.zeros_like(gt)
    pred[7:9, :, :] = 1
    m = match_fragments(gt, pred, method="hungarian")
    assert m == {1: None}
