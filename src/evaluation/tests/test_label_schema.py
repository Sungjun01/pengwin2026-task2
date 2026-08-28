import numpy as np
import pytest

from evaluation.label_schema import (
    BONE_RANGES,
    anatomy_id_from_fragment_id,
    remap_to_anatomy,
)


def test_bone_ranges_match_pengwin_spec():
    assert BONE_RANGES == {
        1: (1, 50),
        2: (51, 100),
        3: (101, 150),
        4: (151, 200),
    }


@pytest.mark.parametrize("frag_id,expected", [
    (1, 1), (50, 1),
    (51, 2), (100, 2),
    (101, 3), (150, 3),
    (151, 4), (200, 4),
    (0, 0),
])
def test_anatomy_id_from_fragment_id(frag_id, expected):
    assert anatomy_id_from_fragment_id(frag_id) == expected


def test_anatomy_id_out_of_range_raises():
    with pytest.raises(ValueError):
        anatomy_id_from_fragment_id(201)


def test_remap_to_anatomy_array():
    lbl = np.array([0, 1, 50, 51, 100, 101, 150, 151, 200], dtype=np.int32)
    anat = remap_to_anatomy(lbl)
    np.testing.assert_array_equal(anat, [0, 1, 1, 2, 2, 3, 3, 4, 4])
