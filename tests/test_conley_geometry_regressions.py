"""Regression tests for the cube geometry of the Conley index complex.

The Conley index of a Morse set S is computed on a cubical complex built at
the depth of S from the index pair X = S union cover(F(S)), A = X minus S
(TreeGrid::relativeComplex). On an adaptive grid X can hold cells deeper
than S, and each of them enters the complex as its ancestor at the depth of
S. The complex bounds were the union of the cell geometries, which stops
short of such an ancestor, so the cubes got the wrong rectangles (C43): the
map was evaluated on rectangles that are not grid boxes, and the indices of
hyperbolic fixed points came out trivial or undefined.

The map is f(x) = x / (2 - x) in every coordinate. Its fixed point 0 is
attracting (f'(0) = 1/2) and 1 is repelling (f'(1) = 2), so a fixed point
with u coordinates equal to 1 has a u-dimensional unstable direction that
preserves orientation. Its Conley index is Z in dimension u with the
identity map (annotation x-1), and 0 in the other dimensions. Uniform grids
at the finer depth give the same indices.
"""

import itertools
import math

import pytest
import CMGDB


def f(x):
    return [xi / (2.0 - xi) for xi in x]


def F(rect):
    return CMGDB.BoxMap(f, rect, padding=False)


def expected_index(point):
    unstable = sum(1 for xi in point if xi == 1.0)
    return ["x-1" if d == unstable else "0" for d in range(len(point) + 1)]


def indices_by_fixed_point(morse_graph, dim):
    result = {}
    for point in itertools.product((0.0, 1.0), repeat=dim):
        result[point] = [
            morse_graph.annotations(v)
            for v in range(morse_graph.num_vertices())
            if any(all(box[d] <= point[d] <= box[dim + d] for d in range(dim))
                   for box in morse_graph.morse_set_boxes(v))
        ]
    return result


def is_subdivision_box(rect, lower, upper):
    # True if rect is a node of the bisection tree of the domain, which
    # splits the coordinates in turn, starting with the first.
    dim = len(lower)
    splits = []
    for d in range(dim):
        width = rect[dim + d] - rect[d]
        ratio = (upper[d] - lower[d]) / width
        k = round(math.log2(ratio))
        if not math.isclose(ratio, 2.0 ** k, rel_tol=1e-9):
            return False
        offset = (rect[d] - lower[d]) / width
        if not math.isclose(offset, round(offset), abs_tol=1e-6):
            return False
        splits.append(k)
    depth = sum(splits)
    return splits == [depth // dim + (d < depth % dim) for d in range(dim)]


# (upper bounds, (subdiv_min, subdiv_max, subdiv_init)); the domains start
# at 0. Before the fix, the repeller had the trivial index in the first case
# and an undefined one in the second; in the third, the saddle (0, 1) and
# the repeller had the trivial index; in the fourth, every index but the
# attractor's was undefined.
CASES = [
    pytest.param([1.25], (4, 8, 0), id="1d-1.25-4-8-0"),
    pytest.param([1.2], (4, 8, 0), id="1d-1.2-4-8-0"),
    pytest.param([1.25, 1.25], (8, 14, 4), id="2d-1.25-8-14-4"),
    pytest.param([1.2, 1.2], (8, 12, 4), id="2d-1.2-8-12-4"),
]


@pytest.mark.parametrize("upper, subdivisions", CASES)
def test_hyperbolic_fixed_points_get_their_conley_index(upper, subdivisions):
    dim = len(upper)
    model = CMGDB.Model(*subdivisions, 10000, [0.0] * dim, upper, F)
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    found = indices_by_fixed_point(morse_graph, dim)
    assert found == {point: [expected_index(point)] for point in found}


@pytest.mark.parametrize(
    "upper, subdivisions",
    # The README model: its indices were right, but 15 of the map
    # evaluations of its Conley phase were not grid boxes.
    CASES + [pytest.param([1.2, 1.2], (6, 10, 4), id="2d-1.2-6-10-4")],
)
def test_conley_phase_evaluates_the_map_on_grid_boxes(upper, subdivisions):
    dim = len(upper)
    lower = [0.0] * dim
    evaluated = []

    def recording_F(rect):
        evaluated.append(list(rect))
        return F(rect)

    model = CMGDB.Model(*subdivisions, 10000, lower, upper, recording_F)
    CMGDB.ComputeConleyMorseGraph(model)
    assert evaluated
    off_grid = [rect for rect in evaluated
                if not is_subdivision_box(rect, lower, upper)]
    assert off_grid == []


@pytest.mark.parametrize("batch", [False, True], ids=["scalar", "batch"])
def test_precomputed_box_map_through_the_conley_phase(batch):
    # PrecomputedBoxMap refuses rectangles off its lattice, so the off-grid
    # cubes made ComputeConleyMorseGraph raise on the README model (C42).
    lower, upper = [0.0, 0.0], [1.2, 1.2]
    F_pre = CMGDB.PrecomputedBoxMap(lambda X: X / (2.0 - X), lower, upper, 10)
    model = CMGDB.Model(6, 10, 4, 10000, lower, upper, F_pre)
    if batch:
        model.set_batch_map(F_pre.batch)
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    found = indices_by_fixed_point(morse_graph, 2)
    assert found == {point: [expected_index(point)] for point in found}
