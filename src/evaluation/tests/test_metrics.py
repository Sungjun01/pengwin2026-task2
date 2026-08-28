import numpy as np
import pytest

from evaluation.metrics import (
    iou, bounding_sphere_diameter_mm, hd95_mm, assd_mm, dice, recognition_quality,
)


def test_iou_identical():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[2:8, 2:8, 2:8] = True
    assert iou(m, m) == 1.0


def test_iou_disjoint():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0:3, :, :] = True
    b[7:10, :, :] = True
    assert iou(a, b) == 0.0


def test_iou_half_overlap_known_value():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[:, :, 0:5] = True   # 10*10*5 = 500
    b[:, :, 2:7] = True   # 10*10*5 = 500
    # intersection: 10*10*3 = 300; union: 10*10*7 = 700
    assert iou(a, b) == pytest.approx(300 / 700)


def test_iou_both_empty_is_zero():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    assert iou(a, b) == 0.0


def test_bounding_sphere_unit_voxel():
    """Single 1 mm^3 voxel -> diameter = 2 * (3/(4pi))^(1/3) ≈ 1.241 mm."""
    m = np.zeros((5, 5, 5), dtype=bool)
    m[2, 2, 2] = True
    d = bounding_sphere_diameter_mm(m, (1.0, 1.0, 1.0))
    expected = 2 * (3 / (4 * np.pi)) ** (1 / 3)
    assert d == pytest.approx(expected, rel=1e-6)


def test_bounding_sphere_anisotropic_spacing():
    """Anisotropic spacing scales voxel volume linearly."""
    m = np.zeros((5, 5, 5), dtype=bool)
    m[2, 2, 2] = True
    d = bounding_sphere_diameter_mm(m, (2.0, 1.0, 1.0))  # voxel vol = 2 mm^3
    expected = 2 * (3 * 2 / (4 * np.pi)) ** (1 / 3)
    assert d == pytest.approx(expected, rel=1e-6)


def test_bounding_sphere_empty_mask_zero():
    m = np.zeros((5, 5, 5), dtype=bool)
    assert bounding_sphere_diameter_mm(m, (1.0, 1.0, 1.0)) == 0.0


def test_hd95_identical_is_zero():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[3:7, 3:7, 3:7] = True
    assert hd95_mm(m, m, spacing=(1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-6)


def test_assd_identical_is_zero():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[3:7, 3:7, 3:7] = True
    assert assd_mm(m, m, spacing=(1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-6)


def test_hd95_disjoint_voxels_returns_voxel_distance():
    """Two single voxels separated by 5 voxels at unit spacing -> 5 mm."""
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros((20, 20, 20), dtype=bool)
    a[10, 10, 5] = True
    b[10, 10, 10] = True
    d = hd95_mm(a, b, spacing=(1.0, 1.0, 1.0))
    assert d == pytest.approx(5.0, abs=1e-6)


def test_hd95_respects_spacing():
    """Same voxel separation, doubled spacing -> doubled mm distance."""
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros((20, 20, 20), dtype=bool)
    a[10, 10, 5] = True
    b[10, 10, 10] = True
    d = hd95_mm(a, b, spacing=(1.0, 1.0, 2.0))
    assert d == pytest.approx(10.0, abs=1e-6)


# --- Dice (LDSC building block) ---


def test_dice_identical():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[2:8, 2:8, 2:8] = True
    assert dice(m, m) == 1.0


def test_dice_disjoint():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0:3, :, :] = True
    b[7:10, :, :] = True
    assert dice(a, b) == 0.0


def test_dice_half_overlap_known_value():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[:, :, 0:5] = True   # 500
    b[:, :, 2:7] = True   # 500
    # intersection 300, sizes 500 + 500 → dice = 2 * 300 / 1000 = 0.6
    assert dice(a, b) == pytest.approx(0.6)


def test_dice_both_empty_is_zero():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    assert dice(a, b) == 0.0


def test_dice_is_one_at_full_overlap():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[1:4, 1:4, 1:4] = True
    assert dice(m, m) == 1.0


# --- Recognition Quality (RQ) ---


def test_rq_perfect_detection():
    """모든 GT 가 high-IoU 로 매칭 → TP only → RQ = 1."""
    assert recognition_quality(tp=10, fp=0, fn=0) == 1.0


def test_rq_all_missed():
    assert recognition_quality(tp=0, fp=0, fn=5) == 0.0


def test_rq_all_spurious():
    assert recognition_quality(tp=0, fp=5, fn=0) == 0.0


def test_rq_panoptic_formula_known_value():
    """RQ = TP / (TP + 0.5 FP + 0.5 FN). TP=4, FP=2, FN=2 → 4 / (4 + 1 + 1) = 4/6."""
    assert recognition_quality(tp=4, fp=2, fn=2) == pytest.approx(4 / 6)


def test_rq_no_predictions_no_gt():
    """0 division 안전 처리."""
    assert recognition_quality(tp=0, fp=0, fn=0) == 0.0
