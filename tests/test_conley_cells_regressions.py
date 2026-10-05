"""Regression tests for ComputeConleyIndexForCells.

- C23/C36: the call releases the GIL, so it may run TreeGrid::cover at the
  same time as another thread; cover must not share scratch state between
  threads.
- C29: a Model without a map, or of another dimension than the Morse
  graph, is refused with ValueError instead of crashing or evaluating the
  map on boxes of the wrong dimension.
- C27: batch_chunk_size bounds the rectangles per call of the batch map,
  as it does in ComputeConleyMorseGraph.
"""

import subprocess
import sys

import pytest

import CMGDB


def product_model(dim=2, subdiv=6):
    """Uniform grid on [0, 1.2]^dim, map x -> x/(2-x) on every axis."""
    def f(x):
        return [x[d] / (2.0 - x[d]) for d in range(dim)]

    def F(rect):
        return CMGDB.BoxMap(f, rect)

    return CMGDB.Model(subdiv, subdiv, subdiv, 10000,
                       [0.0] * dim, [1.2] * dim, F)


def cubic(x):
    return 1.5 * x - 0.5 * x * x * x


def cubic_model(subdiv=8):
    """Uniform grid on [-1.5, 1.5]^2, map g(x) = 1.5x - 0.5x^3 on both axes.

    g maps [-1.5, 1.5] into [-1, 1], so the whole grid is a valid cell set.
    The scalar map and cubic_batch agree exactly (products and sums only).
    """
    def f(x):
        return [cubic(x[0]), cubic(x[1])]

    def F(rect):
        return CMGDB.BoxMap(f, rect)

    return CMGDB.Model(subdiv, subdiv, subdiv, 10000,
                       [-1.5, -1.5], [1.5, 1.5], F)


def cubic_batch(rects):
    return CMGDB.BoxMapBatch(cubic, rects)


# Runs in a subprocess, so that a race that corrupts the heap or trips the
# abort() guards of TreeGrid::coverAccept fails this test instead of killing
# the test session.
CONCURRENT_COVER_SCRIPT = r"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import CMGDB

SAMPLES = np.linspace(0.0, 1.0, 4)


def g(x):
    return 1.5 * x - 0.5 * x ** 3


def F_batch(rects):
    rects = np.asarray(rects, dtype=float)
    lower, upper = rects[:, :2], rects[:, 2:]
    points = lower[:, :, None] + SAMPLES * (upper - lower)[:, :, None]
    images = g(points)
    pad = 0.5 * (upper - lower)
    return np.hstack([images.min(axis=2) - pad, images.max(axis=2) + pad])


def F(rect):
    return F_batch([rect])[0].tolist()


# g maps [-1.5, 1.5] into [-1, 1], so the whole grid is a valid cell set;
# it makes every call cover thousands of images.
model = CMGDB.Model(12, 12, 12, 10000, [-1.5, -1.5], [1.5, 1.5], F)
model.set_batch_map(F_batch)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model, cache_map_graph=False)
cell_sets = [list(range(map_graph.num_vertices()))]
cell_sets += [morse_graph.morse_set(v) for v in range(morse_graph.num_vertices())]
expected = [CMGDB.ComputeConleyIndexForCells(model, morse_graph, cells)
            for cells in cell_sets]


def sweep(_):
    return [CMGDB.ComputeConleyIndexForCells(model, morse_graph, cells)
            for cells in cell_sets]


# Several threads inside ComputeConleyIndexForCells at once.
with ThreadPoolExecutor(4) as pool:
    for result in pool.map(sweep, range(16)):
        assert result == expected

# One thread inside ComputeConleyIndexForCells while this one covers with
# the GIL held, through the adjacencies of the lazy map graph.
queried = range(0, map_graph.num_vertices(), 7)
adjacencies = {c: sorted(map_graph.adjacencies(c)) for c in queried}
with ThreadPoolExecutor(1) as pool:
    worker = pool.submit(sweep, 0)
    passes = 0
    while not worker.done():
        for c in queried:
            assert sorted(map_graph.adjacencies(c)) == adjacencies[c]
        passes += 1
    assert worker.result() == expected
assert passes > 0
print("ok")
"""


def test_concurrent_calls_do_not_share_cover_state():
    completed = subprocess.run(
        [sys.executable, "-c", CONCURRENT_COVER_SCRIPT],
        capture_output=True, text=True, timeout=600,
    )
    status, output = completed.returncode, completed.stdout + completed.stderr
    assert status == 0, f"exit status {status}\n{output}"
    assert completed.stdout.strip() == "ok", output


def test_model_without_map_is_refused():
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(product_model())
    with pytest.raises(ValueError, match="requires a Model with a map"):
        CMGDB.ComputeConleyIndexForCells(
            CMGDB.Model(), morse_graph, morse_graph.morse_set(0))


def test_model_of_another_dimension_is_refused():
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(product_model(dim=2))
    # The identity box map accepts boxes of any dimension, so nothing but
    # the dimension check stops the 1D model from running on 2D boxes.
    model = CMGDB.Model(6, 6, 6, 10000, [0.0], [1.2], lambda rect: rect)
    with pytest.raises(ValueError, match="dimension 1.*dimension 2"):
        CMGDB.ComputeConleyIndexForCells(
            model, morse_graph, morse_graph.morse_set(0))


def test_batch_chunk_size_bounds_rows_per_batch_call():
    model = cubic_model()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    cells = list(range(map_graph.num_vertices()))
    expected = CMGDB.ComputeConleyIndexForCells(model, morse_graph, cells)
    rows = []

    def F_batch(rects):
        rows.append(len(rects))
        return cubic_batch(rects)

    model.set_batch_map(F_batch)
    assert CMGDB.ComputeConleyIndexForCells(model, morse_graph, cells) == expected
    assert max(rows) == len(cells)
    rows.clear()
    index = CMGDB.ComputeConleyIndexForCells(
        model, morse_graph, cells, batch_chunk_size=8)
    assert index == expected
    assert rows and max(rows) <= 8
