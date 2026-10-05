"""Regression tests for the precomputed box maps: the PrecomputedBoxMap class
and the grid-layout factories of CMGDB.precomputed_grid."""

import numpy as np
import pytest
import CMGDB
from CMGDB import precomputed_grid


# Domains on which lower + cells*side, the last lattice node, rounds above
# (1.2000000000000002) or below (0.2999999999999998) the upper bound
ROUNDING_DOMAINS = [([-1.0, -1.0], [1.2, 1.2]), ([-3.0, -3.0], [0.3, 0.3])]


def domain_checked_map(lower, upper, sampled):
    """A map defined on the closed domain only, like a bounded interpolant:
    evaluating it outside raises. Records the largest point of each call."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def f(X):
        X = np.asarray(X, dtype=float)
        if np.any(X < lower) or np.any(X > upper):
            raise ValueError("map evaluated outside the domain")
        sampled.append(X.max(axis=0))
        return 0.5 * X
    return f


@pytest.mark.parametrize("lower, upper", ROUNDING_DOMAINS)
@pytest.mark.parametrize("mode", ["corners", "center"])
def test_class_lattice_ends_at_upper_bound(lower, upper, mode):
    # C47: TreeGrid puts the upper face of the boundary boxes at upper_bounds
    # exactly, and live BoxMap samples f there; the table must too
    sampled = []
    CMGDB.PrecomputedBoxMap(domain_checked_map(lower, upper, sampled),
                            lower, upper, 10, mode=mode)
    assert np.array_equal(np.max(sampled, axis=0), upper)


@pytest.mark.parametrize("lower, upper", ROUNDING_DOMAINS)
@pytest.mark.parametrize("layout", ["adaptive", "uniform"])
@pytest.mark.parametrize("eval_mode", ["corners", "center", "random"])
def test_factory_lattice_ends_at_upper_bound(lower, upper, layout, eval_mode):
    # C47, in the fork's precompute_corner_grid
    sampled = []
    CMGDB.make_precomputed_box_map(domain_checked_map(lower, upper, sampled),
                                   lower, upper, subdiv_max=10, mode=layout,
                                   eval_mode=eval_mode)
    assert np.array_equal(np.max(sampled, axis=0), upper)


def test_corner_grid_ends_at_upper_bound_on_refined_axes_only():
    # C47; an axis with a single node keeps it at the lower bound
    grid, _ = precomputed_grid.precompute_corner_grid(
        lambda X: X, lower_bounds=[-1.0, -1.0], upper_bounds=[1.2, 1.2],
        corners_per_axis=[1, 3])
    assert np.array_equal(grid[0, :, 0], [-1.0, -1.0, -1.0])
    assert grid[0, 0, 1] == -1.0 and grid[0, -1, 1] == 1.2


def test_upper_face_box_matches_live_box_map():
    # C47: for a map with sqrt(ub - x), the overshooting node gave NaN images
    # on every box of the upper face
    lower, upper = [-1.0, -1.0], [1.2, 1.2]

    def f_vec(X):
        return np.column_stack([1.2 - 0.8 * np.sqrt(1.2 - X[:, 0]), 0.5 * X[:, 1]])

    def f_scalar(x):
        return list(f_vec(np.array([x]))[0])

    F = CMGDB.PrecomputedBoxMap(f_vec, lower, upper, 10)
    assert not np.isnan(F._table).any()
    side = F._finest_box_side
    rect = [upper[0] - side[0], lower[1], upper[0], lower[1] + side[1]]
    assert np.allclose(F(rect), CMGDB.BoxMap(f_scalar, rect), rtol=0, atol=1e-12)


# Rectangles that reach outside [0, 1]^2 by a whole cell (side 1/16 at
# subdiv_max=8) or more, or have a NaN bound
SIDE = 1.0 / 16.0
OUTSIDE_RECTS = [
    [-SIDE, 0.0, SIDE, SIDE],             # crosses the lower face
    [0.5, 0.5, 1.0 + 4 * SIDE, 1.0],      # crosses the upper face
    [-1.0, -1.0, 2.0, 2.0],               # contains the whole domain
    [1.0, 1.0, 1.0 + SIDE, 1.0 + SIDE],   # entirely outside
    [np.nan, 0.0, SIDE, SIDE],
]


def double(X):
    return 2.0 * np.asarray(X, dtype=float)


@pytest.mark.parametrize("rect", OUTSIDE_RECTS)
def test_class_rejects_box_outside_domain(rect):
    # C48: the lattice indices were clipped to the domain, so the image of a
    # smaller box came back without an error
    F = CMGDB.PrecomputedBoxMap(double, [0.0, 0.0], [1.0, 1.0], 8)
    with pytest.raises(ValueError, match="outside"):
        F(rect)
    with pytest.raises(ValueError, match="outside"):
        F.batch([[0.0, 0.0, SIDE, SIDE], rect])


@pytest.mark.parametrize("layout", ["adaptive", "uniform"])
@pytest.mark.parametrize("rect", OUTSIDE_RECTS)
def test_factories_reject_box_outside_domain(layout, rect):
    # C48, in the fork's factories
    box_map = CMGDB.make_precomputed_box_map(double, [0.0, 0.0], [1.0, 1.0],
                                             subdiv_max=8, mode=layout)
    with pytest.raises(ValueError, match="outside"):
        box_map(rect)
    with pytest.raises(ValueError, match="outside"):
        box_map.batch([[0.0, 0.0, SIDE, SIDE], rect])


def test_boxes_on_the_domain_faces_still_map():
    F = CMGDB.PrecomputedBoxMap(double, [0.0, 0.0], [1.0, 1.0], 8)
    rects = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, SIDE, SIDE],
             [1.0 - SIDE, 1.0 - SIDE, 1.0, 1.0]]
    expected = [double(r).tolist() for r in rects]
    assert [F(r) for r in rects] == expected
    assert F.batch(rects).tolist() == expected
