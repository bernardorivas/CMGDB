"""Regression tests for ComputeConleyIndexForCells.

- C23/C36: the call releases the GIL, so it may run TreeGrid::cover at the
  same time as another thread; cover must not share scratch state between
  threads.
"""

import subprocess
import sys

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
