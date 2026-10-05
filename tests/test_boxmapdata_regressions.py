"""Regression tests for the spatial index of BoxMapData: it must select the
same points as the linear scan of BoxMapDataLinear."""

import numpy as np
import pytest
import CMGDB


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(5000, 2))
    return X, 0.5 * X


INF, NAN = np.inf, np.nan
NONFINITE_OR_HUGE_RECTS = [
    [0.5, 0.5, INF, INF],
    [0.5, 0.0, 1e300, 1.0],
    [-INF, -INF, INF, INF],
    [0.0, 0.0, 1e20, 1e20],
    [-1e300, 0.3, 0.6, 0.9],
    [NAN, 0.0, 1.0, 1.0],
    [0.0, 0.0, NAN, 1.0],
]


# The bin of a bound was cast to int64 before it was clipped. The cast is
# undefined for inf, NaN and values of 2^63 bins or more: x86-64 gives
# INT64_MIN, which clipped to bin 0 and lost the points above it, while
# arm64 saturates. NumPy warns about the cast on both, so the warning is
# made an error to catch the cast on any machine.
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("rect", NONFINITE_OR_HUGE_RECTS)
def test_index_matches_linear_scan_on_nonfinite_or_huge_bounds(data, rect):
    # C41
    X, Y = data
    indexed = CMGDB.BoxMapData(X, Y)
    assert indexed._use_index
    expected = CMGDB.BoxMapDataLinear(X, Y).map_points(rect)
    assert np.array_equal(indexed.map_points(rect), expected)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_index_matches_linear_scan_on_tiny_spread_coordinate():
    # C41: with a coordinate of spread 1e-20 (zero up to roundoff, as on an
    # invariant subspace) the bins are so thin that the bounds of ordinary
    # grid boxes lie more than 2^63 bins away
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 5000), rng.uniform(0, 1e-20, 5000)])
    Y = 0.5 * X
    indexed = CMGDB.BoxMapData(X, Y)
    linear = CMGDB.BoxMapDataLinear(X, Y)
    for x0 in np.linspace(-1.0, 0.75, 8):
        for y0 in np.linspace(-1.0, 0.75, 8):
            rect = [x0, y0, x0 + 0.25, y0 + 0.25]
            assert np.array_equal(indexed.map_points(rect), linear.map_points(rect))
            assert indexed.compute(rect) == linear.compute(rect)
