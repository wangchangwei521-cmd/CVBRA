from __future__ import annotations

import math
from itertools import pairwise

import pytest
from scripts.prepare_nvd_real_snow_cvbra_v1 import (
    _temporal_blocks,
    evenly_spaced_indices,
    rotated_hbb,
)


def test_evenly_spaced_indices_are_unique_and_endpoint_complete() -> None:
    indices = evenly_spaced_indices(0, 2479, 900)

    assert len(indices) == 900
    assert len(set(indices)) == 900
    assert indices[0] == 0
    assert indices[-1] == 2479
    assert all(left < right for left, right in pairwise(indices))


def test_evenly_spaced_indices_reject_invalid_oversampling() -> None:
    with pytest.raises(ValueError, match="invalid deterministic frame-selection"):
        evenly_spaced_indices(0, 2, 4)


def test_rotated_hbb_preserves_zero_degree_box() -> None:
    result = rotated_hbb(10.0, 20.0, 50.0, 80.0, 0.0, 100, 100)

    assert result == pytest.approx((10.0, 20.0, 50.0, 80.0))


def test_rotated_hbb_swaps_extents_at_ninety_degrees() -> None:
    result = rotated_hbb(30.0, 40.0, 70.0, 60.0, 90.0, 100, 100)

    assert result == pytest.approx((40.0, 30.0, 60.0, 70.0))


def test_rotated_hbb_clips_enclosing_rectangle() -> None:
    result = rotated_hbb(0.0, 0.0, 20.0, 20.0, 45.0, 100, 100)

    assert result is not None
    assert result[0] == 0.0
    assert result[1] == 0.0
    expected_extent = 10.0 + 10.0 * math.sqrt(2.0)
    assert result[2] == pytest.approx(expected_extent)
    assert result[3] == pytest.approx(expected_extent)


def test_temporal_blocks_are_contiguous_complete_and_nearly_equal() -> None:
    indices = tuple(range(2027))
    blocks = _temporal_blocks(indices, 32)
    grouped = {
        block: [frame for frame, observed in blocks.items() if observed == block]
        for block in range(32)
    }

    assert len(blocks) == 2027
    assert max(map(len, grouped.values())) - min(map(len, grouped.values())) == 1
    assert all(frames == list(range(frames[0], frames[-1] + 1)) for frames in grouped.values())
