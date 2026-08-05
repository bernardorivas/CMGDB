"""Non-corner evaluation modes for the precomputed box maps.

``CMGDB.ComputeBoxMap.BoxMap`` has always offered ``corners``, ``center``, and
``random`` sampling. Precomputation originally supported only ``corners``, so
choosing a precomputed backend silently forfeited the other two. These tests
pin the generalization, and in particular the property that makes it possible:
evaluation offsets are dyadic rationals, so they land on a shared lattice for
every box depth at once.
"""

import itertools

import numpy as np
import pytest

import CMGDB
import CMGDB.PrecomputedBoxMap as P

LOWER = [-1.0, -1.0]
UPPER = [1.0, 1.0]


def vector_map(points):
    points = np.asarray(points, dtype=np.float64)
    return np.stack(
        [
            np.sin(points[:, 0]) + 0.3 * points[:, 1],
            np.cos(points[:, 1]) - 0.2 * points[:, 0],
        ],
        axis=1,
    )


def _make(**kwargs):
    kwargs.setdefault("subdiv_max", 10)
    kwargs.setdefault("mode", "adaptive")
    return CMGDB.make_precomputed_box_map(vector_map, LOWER, UPPER, **kwargs)


# Uniform grids serve finest-level boxes only (subdiv_init == min == max), so
# their rectangles are exactly one box wide. Adaptive grids are queried at
# every depth.
FINEST = [-1.0, -1.0, -0.9375, -0.9375]
COARSE_RECTS = [FINEST, [-1.0, -1.0, -0.75, -0.75], [0.0, 0.0, 1.0, 1.0], LOWER + UPPER]


@pytest.mark.parametrize("mode", ["adaptive", "uniform"])
def test_corner_mode_matches_direct_corner_evaluation(mode):
    """Corner mode is unchanged by the generalization: depth 0, same lattice."""
    rects = [FINEST, [0.0, 0.0, 0.0625, 0.0625]]
    if mode == "adaptive":
        rects.append(LOWER + UPPER)
    box_map = _make(mode=mode, padding=False, eval_mode="corners")
    for rect in rects:
        corners = np.array(
            list(itertools.product([rect[0], rect[2]], [rect[1], rect[3]]))
        )
        values = vector_map(corners)
        expected = np.concatenate([values.min(axis=0), values.max(axis=0)])
        np.testing.assert_allclose(box_map(rect), expected, atol=1e-12)


@pytest.mark.parametrize("rect", COARSE_RECTS)
def test_center_mode_hits_the_exact_midpoint_at_every_depth(rect):
    """The finest boxes are the hard case.

    A finest box's center sits halfway between two corner-lattice nodes, so it
    is unreachable from the corner table. One extra refinement level per axis
    makes every center a node, at every depth.
    """
    box_map = _make(padding=False, eval_mode="center")
    center = np.array([[(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]])
    value = vector_map(center)[0]
    size = np.array(rect[2:]) - np.array(rect[:2])
    # padding is forced on for center; see test_center_mode_forces_padding.
    expected = np.concatenate([value - size, value + size])
    np.testing.assert_allclose(box_map(rect), expected, atol=1e-12)


def test_center_mode_forces_padding():
    """A single sample gives a degenerate image, so it must be padded."""
    box_map = _make(mode="uniform", padding=False, eval_mode="center")
    out = np.asarray(box_map([0.0, 0.0, 0.0625, 0.0625]))
    assert not np.allclose(out[:2], out[2:])


@pytest.mark.parametrize(
    "rect", [FINEST, [0.0, 0.0, 0.5, 0.5], LOWER + UPPER]
)
def test_random_mode_evaluates_at_the_declared_offsets(rect):
    numerators, depth = P.evaluation_offsets(
        "random", 2, num_pts=6, sample_depth=3, seed=0
    )
    box_map = _make(
        padding=False, eval_mode="random", num_pts=6, sample_depth=3, seed=0
    )
    lower = np.array(rect[:2])
    upper = np.array(rect[2:])
    values = vector_map(lower + (numerators / 2**depth) * (upper - lower))
    expected = np.concatenate([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_allclose(box_map(rect), expected, atol=1e-12)


def test_random_offsets_are_dyadic_and_inside_the_box():
    numerators, depth = P.evaluation_offsets(
        "random", 3, num_pts=8, sample_depth=4, seed=11
    )
    assert numerators.shape == (8, 3)
    assert depth == 4
    assert numerators.min() >= 0
    assert numerators.max() <= 2**4


def test_random_mode_is_reproducible_under_a_fixed_seed():
    """Upstream ``BoxMap(mode='random')`` redraws on every call, which makes it
    a non-deterministic function of the rectangle. Fixed offsets do not."""
    rect = [0.0, 0.0, 0.25, 0.25]
    same_a = _make(eval_mode="random", num_pts=5, seed=7)
    same_b = _make(eval_mode="random", num_pts=5, seed=7)
    other = _make(eval_mode="random", num_pts=5, seed=9)
    assert same_a(rect) == same_b(rect)
    assert same_a(rect) != other(rect)


@pytest.mark.parametrize("mode", ["adaptive", "uniform"])
@pytest.mark.parametrize("eval_mode", ["corners", "center", "random"])
def test_scalar_and_batch_paths_agree(mode, eval_mode):
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        LOWER,
        UPPER,
        subdiv_max=8,
        mode=mode,
        padding=True,
        eval_mode=eval_mode,
        num_pts=4,
        sample_depth=2,
        seed=3,
    )
    rng = np.random.default_rng(1)
    rects = []
    for _ in range(25):
        lo = rng.integers(0, 16, size=2) / 8.0 - 1.0
        rects.append([lo[0], lo[1], lo[0] + 0.125, lo[1] + 0.125])
    np.testing.assert_allclose(
        np.asarray(box_map.batch(rects)),
        np.asarray([box_map(rect) for rect in rects]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("eval_mode", "expected_depth"),
    [("corners", 0), ("center", 1)],
)
def test_refinement_depth_is_the_documented_table_cost(eval_mode, expected_depth):
    """Depth d multiplies table size by 2**(dim * d); corners pay nothing."""
    assert P.evaluation_offsets(eval_mode, 3)[1] == expected_depth


def test_random_refinement_depth_follows_sample_depth():
    assert P.evaluation_offsets("random", 3, sample_depth=2)[1] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eval_mode": "bogus"}, "eval_mode"),
        ({"eval_mode": "random", "num_pts": 0}, "num_pts"),
        ({"eval_mode": "random", "sample_depth": 0}, "sample_depth"),
    ],
)
def test_invalid_sampling_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _make(subdiv_max=8, **kwargs)
