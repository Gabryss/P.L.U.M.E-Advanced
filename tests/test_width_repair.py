"""A shortened/repeated station must not create a new width discontinuity."""

import numpy as np
import pytest

from plume_advanced.stages.network_acceptance import limit_width_gradient


@pytest.mark.parametrize("seed", [0, 1, 3, 17, 42, 4294967295])
@pytest.mark.parametrize("gradient", [0.05, 0.6, 2.0])
def test_width_projection_is_bounded_nonexpanding_and_idempotent(seed, gradient):
    rng = np.random.default_rng(seed)
    arc = np.r_[0, np.cumsum(rng.uniform(0.00001, 4, 300))]
    widths = rng.uniform(1, 80, len(arc))
    actual = limit_width_gradient(widths, arc, gradient)
    assert np.all(actual > 0) and np.all(actual <= widths)
    assert np.max(np.abs(np.diff(actual)) / np.diff(arc)) <= gradient + 1e-7
    np.testing.assert_allclose(limit_width_gradient(actual, arc, gradient), actual, atol=1e-10)
    # The closed-form minimum of all cone constraints is an independent oracle.
    expected = np.min(widths[None, :] + gradient * np.abs(arc[:, None] - arc[None, :]), axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_duplicate_stations_receive_identical_width():
    actual = limit_width_gradient(np.array([5.0, 2.0, 4.0]), np.array([0.0, 0.0, 1.0]), 0.6)
    np.testing.assert_allclose(actual, [2, 2, 2.6])


@pytest.mark.parametrize(
    "widths,arc,gradient",
    [
        ([1, 2], [0], 0.6),
        ([1, np.nan], [0, 1], 0.6),
        ([1, 2], [1, 0], 0.6),
        ([0, 2], [0, 1], 0.6),
        ([1, 2], [0, 1], 0),
    ],
)
def test_invalid_width_constraints_fail_clearly(widths, arc, gradient):
    with pytest.raises(ValueError):
        limit_width_gradient(np.asarray(widths), np.asarray(arc), gradient)
