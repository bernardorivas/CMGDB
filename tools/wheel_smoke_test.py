"""Post-build check that a wheel is importable and carries the fork's features.

Run by cibuildwheel's test-command against the freshly built wheel, in a
throwaway environment with no access to the source tree.
"""

import faulthandler
import math
import sys

# A crash in the extension is otherwise a bare exit code in the CI log.
faulthandler.enable()


def step(message):
    print(f"[smoke] {message}", flush=True)


step("importing CMGDB")
import CMGDB  # noqa: E402

# The pure-Python layer that consumers import must ship alongside the extension.
from CMGDB.PrecomputedBoxMap import precompute_corner_grid  # noqa: F401
from CMGDB import cmgdb_roa, morse_graph_parser, morse_lattice  # noqa: F401

for name in ("Model", "BoxMap", "ComputeMorseGraph", "ComputeConleyMorseGraph"):
    assert hasattr(CMGDB, name), f"missing upstream entry point {name}"

for name in ("ComputeMorseSetReachability", "MorseDirectedPathCells"):
    assert hasattr(CMGDB, name), f"missing fork entry point {name}"


def f(x):
    # Planar Leslie population map, as in the CMGDB primer notebook.
    th1, th2 = 20.0, 20.0
    return [(th1 * x[0] + th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]


def box_map(rect):
    x0_min, x1_min, x0_max, x1_max = rect
    ys = [
        f([x0_min, x1_min]),
        f([x0_max, x1_min]),
        f([x0_min, x1_max]),
        f([x0_max, x1_max]),
    ]
    return [
        min(y[0] for y in ys),
        min(y[1] for y in ys),
        max(y[0] for y in ys),
        max(y[1] for y in ys),
    ]


step(f"evaluating the box map: {box_map([1.0, 1.0, 2.0, 2.0])}")

step("building the model")
model = CMGDB.Model(14, 14, 14, 10000, [-0.001, -0.001], [90.0, 70.0], box_map)

step("computing the Morse graph")
morse_graph, _ = CMGDB.ComputeMorseGraph(model)

n = morse_graph.num_vertices()
assert n > 0, "Morse graph came back empty"
step(f"{CMGDB.__file__}: Morse graph with {n} vertices")
sys.exit(0)
