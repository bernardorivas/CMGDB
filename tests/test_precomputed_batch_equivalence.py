"""The vectorized batch path must equal the scalar path bit for bit.

`PrecomputedBoxMap.batch` used to be `[self._lookup(r) for r in rects]`. It is
now vectorized over the whole chunk, which is where nearly all of the 2-D
`ComputeConleyMorseGraph` speedup comes from. Since CMGDB feeds every adjacency
query through this interface, any divergence between the two paths silently
changes the computed Morse decomposition. These tests pin them together at
`rtol=0, atol=0`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from CMGDB.PrecomputedBoxMap import (
    make_adaptive_precomputed_box_map,
    make_uniform_precomputed_box_map,
)


def nonlinear_map(dim: int):
    """A deliberately non-monotone map, so corner min/max is not the endpoints."""

    def f(points):
        P = np.asarray(points, dtype=np.float64)
        out = np.empty_like(P)
        out[:, 0] = np.sin(3.0 * P[:, 0]) + 0.5 * P[:, -1] ** 2
        for k in range(1, dim):
            out[:, k] = np.cos(2.0 * P[:, k]) - 0.25 * P[:, 0]
        return out

    return f


def random_rects(rng, lower, upper, n_per_axis, dim, m, *, snap):
    """Rectangles on the lattice (snap=True) or arbitrary sub-boxes (snap=False)."""
    side = (np.asarray(upper) - np.asarray(lower)) / n_per_axis
    if snap:
        i = rng.integers(0, n_per_axis, size=(m, dim))
        lo = np.asarray(lower) + i * side
        hi = lo + side
    else:
        lo = rng.uniform(lower, upper, size=(m, dim))
        hi = lo + rng.uniform(0.0, 1.0, size=(m, dim)) * side
        hi = np.minimum(hi, upper)
    return np.concatenate([lo, hi], axis=1).tolist()


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("snap", [True, False])
def test_adaptive_batch_matches_scalar(dim, padding, snap):
    lower = [0.0] * dim
    upper = [1.0] * dim
    subdiv_max = 4 * dim  # keeps the table small in every dimension
    bm = make_adaptive_precomputed_box_map(
        nonlinear_map(dim),
        lower_bounds=lower,
        upper_bounds=upper,
        subdiv_max=subdiv_max,
        padding=padding,
    )
    n_per_axis = 2 ** -(-subdiv_max // dim)
    rng = np.random.default_rng(0)
    rects = random_rects(rng, lower, upper, n_per_axis, dim, 500, snap=snap)

    scalar = np.array([bm(r) for r in rects])
    batched = np.array(bm.batch(rects))
    assert batched.shape == scalar.shape
    assert np.array_equal(batched, scalar)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("snap", [True, False])
def test_uniform_batch_matches_scalar(dim, padding, snap):
    lower = [0.0] * dim
    upper = [1.0] * dim
    subdiv_max = 4 * dim  # must be divisible by dim for the uniform backend
    bm = make_uniform_precomputed_box_map(
        nonlinear_map(dim),
        lower_bounds=lower,
        upper_bounds=upper,
        subdiv_max=subdiv_max,
        padding=padding,
    )
    n_per_axis = 2 ** (subdiv_max // dim)
    rng = np.random.default_rng(1)
    rects = random_rects(rng, lower, upper, n_per_axis, dim, 500, snap=snap)

    scalar = np.array([bm(r) for r in rects])
    batched = np.array(bm.batch(rects))
    assert batched.shape == scalar.shape
    assert np.array_equal(batched, scalar)


@pytest.mark.parametrize(
    "factory", [make_adaptive_precomputed_box_map, make_uniform_precomputed_box_map]
)
def test_empty_batch_returns_empty(factory):
    bm = factory(
        nonlinear_map(2),
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=8,
    )
    assert bm.batch([]) == []


@pytest.mark.parametrize(
    "factory", [make_adaptive_precomputed_box_map, make_uniform_precomputed_box_map]
)
def test_batch_boundary_rects(factory):
    """Rectangles pinned to the domain corners exercise the index clipping."""
    lower = [0.0, 0.0]
    upper = [1.0, 1.0]
    bm = factory(
        nonlinear_map(2), lower_bounds=lower, upper_bounds=upper, subdiv_max=8
    )
    side = 1.0 / 16.0
    rects = []
    for lo in itertools.product([0.0, 1.0 - side], repeat=2):
        rects.append([lo[0], lo[1], lo[0] + side, lo[1] + side])
    # Degenerate rectangle and one that sits exactly on the upper bound.
    rects.append([1.0, 1.0, 1.0, 1.0])
    rects.append([0.0, 0.0, 1.0, 1.0])

    scalar = np.array([bm(r) for r in rects])
    batched = np.array(bm.batch(rects))
    assert np.array_equal(batched, scalar)


def test_conley_morse_graph_identical_with_and_without_batch():
    """End to end: installing the batch map must not change the computed graph.

    The unit tests above pin the two lookup paths together. This one pins the
    thing that actually matters -- that CMGDB's Morse decomposition and Conley
    indices come out identical whether or not `set_batch_map` is installed.
    """
    import CMGDB

    lower, upper = [0.0, 0.0], [1.0, 1.0]
    subdiv_min, subdiv_max, subdiv_init = 10, 12, 8

    def build():
        return make_adaptive_precomputed_box_map(
            nonlinear_map(2),
            lower_bounds=lower,
            upper_bounds=upper,
            subdiv_max=subdiv_max,
            padding=True,
        )

    def compute(use_batch):
        bm = build()
        model = CMGDB.Model(
            subdiv_min, subdiv_max, subdiv_init, 10000, lower, upper, bm
        )
        if use_batch:
            model.set_batch_map(bm.batch)
        mg, _ = CMGDB.ComputeConleyMorseGraph(model)
        vertices = sorted(mg.vertices())
        annotations = [tuple(str(a) for a in mg.annotations(v)) for v in vertices]
        edges = sorted((int(u), int(w)) for u, w in mg.edges())
        return annotations, edges

    scalar = compute(False)
    batched = compute(True)
    assert scalar == batched
    assert scalar[0], "expected a non-empty Morse graph for this fixture"


def test_batch_matches_scalar_on_numpy_input():
    """CMGDB may hand over a NumPy array rather than a list of lists."""
    bm = make_adaptive_precomputed_box_map(
        nonlinear_map(2),
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=8,
    )
    rng = np.random.default_rng(2)
    rects = np.array(random_rects(rng, [0.0, 0.0], [1.0, 1.0], 16, 2, 64, snap=True))
    scalar = np.array([bm(r) for r in rects])
    batched = np.array(bm.batch(rects))
    assert np.array_equal(batched, scalar)
