"""Regression tests for the CHomP Conley-index computation.

C37: a Morse set away from the boundary of the phase space got an undefined
Conley index (``annotations(v) == []``) when the whole image of one of its
exit boxes lies beyond an upper face of the phase space. The cover clips such
an image to nothing (it clamps one below a lower face onto the bottom layer
of the complex, which spans the boxes of the index pair), which leaves a
fiber of the graph of the map without its relative part. The index is now
recomputed with the images that miss the phase space projected onto it; the
expected values below are the indices of the hyperbolic fixed points, which
the same boxes on a larger phase space also give.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import CMGDB


def _morse_sets(f, lower, upper, subdiv, padding=False):
    model = CMGDB.Model(subdiv, subdiv, subdiv, 10000, lower, upper,
                        lambda rect: CMGDB.BoxMap(f, rect, padding=padding))
    morse_graph = CMGDB.ComputeConleyMorseGraphOnly(model)
    return [(np.array(morse_graph.morse_set_boxes(v)), morse_graph.annotations(v))
            for v in range(morse_graph.num_vertices())]


def _set_at(point, sets):
    """The boxes and annotations of the Morse set whose boxes contain `point`."""
    dim = len(point)
    for boxes, annotations in sets:
        if any(np.all(box[:dim] <= point) and np.all(point <= box[dim:]) for box in boxes):
            return boxes, annotations
    raise AssertionError(f"no Morse set contains {point}")


def _index_at(point, sets):
    return _set_at(point, sets)[1]


@pytest.mark.parametrize("slope, expected", [(10.0, ["0", "x-1"]), (-10.0, ["0", "x+1"])])
def test_repeller_with_exit_images_outside_the_phase_space(slope, expected):
    # f(x) = slope * x on [-1, 1] with 16 boxes. The Morse set is the two
    # boxes around 0. The exit boxes next to it map onto [1.25, 2.5] and
    # [-2.5, -1.25], outside the phase space: the cover of the first image
    # is empty, and the second is clamped onto the box [-1, -0.875].
    f = lambda x: [slope * x[0]]
    assert _index_at([0.0], _morse_sets(f, [-1.0], [1.0], 4)) == expected
    # The same boxes on [-2, 2]: no image of a box next to the Morse set
    # lies wholly outside the phase space.
    assert _index_at([0.0], _morse_sets(f, [-2.0], [2.0], 5)) == expected


@pytest.mark.parametrize("linear, expected", [
    ((10.0, 0.5), ["0", "x-1", "0"]),
    ((-10.0, 0.5), ["0", "x+1", "0"]),
    ((10.0, 10.0), ["0", "0", "x-1"]),
    ((10.0, -8.0), ["0", "0", "x+1"]),
])
def test_hyperbolic_fixed_point_with_exit_images_outside_the_phase_space(linear, expected):
    a, b = linear
    f = lambda x: [a * x[0], b * x[1]]
    assert _index_at([0.0, 0.0], _morse_sets(f, [-1.0, -1.0], [1.0, 1.0], 6)) == expected


def test_reported_interior_source():
    # The reported case: f(x) = A x + B tanh(C x), padded box map on
    # [-3, 3]^2 at subdivision 10. The origin is a repelling fixed point
    # (eigenvalues -1.30 and -6.42 of Df(0) = A + B C) inside a Morse set of
    # 56 boxes that stays off the boundary.
    A = np.array([[-1.2528613903720396, 0.5412688503713783],
                  [0.057072964981257385, -0.3418598437950228]])
    B = np.array([[-0.5526637075920041, 0.6331197444475793],
                  [1.2519464007702688, -0.6204529038332427]])
    C = np.array([[-0.4999102341314765, -3.8998322676201473],
                  [0.9982946143968527, 3.4657250827141843]])
    f = lambda x: list(A @ np.asarray(x) + B @ np.tanh(C @ np.asarray(x)))
    boxes, annotations = _set_at([0.0, 0.0], _morse_sets(f, [-3.0, -3.0], [3.0, 3.0], 10,
                                                         padding=True))
    assert not np.any(np.isclose(np.abs(boxes), 3.0))
    assert annotations == ["0", "0", "x-1"]
    # The same box size on [-6, 6]^2, where the exit images stay inside.
    assert _index_at([0.0, 0.0], _morse_sets(f, [-6.0, -6.0], [6.0, 6.0], 12,
                                             padding=True)) == ["0", "0", "x-1"]


def test_morse_set_on_the_boundary_stays_undefined():
    # The saddle at the origin of (10 x, y / 2) on [-1, 1] x [0, 1] lies on
    # the boundary y = 0. For a Morse set on the boundary, projecting the
    # exit images onto the phase space is not justified (a projected image
    # can land in the set), so its index is deliberately left undefined.
    f = lambda x: [10.0 * x[0], 0.5 * x[1]]
    assert _index_at([0.0, 0.0], _morse_sets(f, [-1.0, 0.0], [1.0, 1.0], 6)) == []


def test_undefined_index_is_silent():
    # The C++ core prints only under CMG_VERBOSE. ComputeConleyIndex printed
    # "Problem computing conley index" and the reason to stdout whenever it
    # returned an undefined index. The cases are 1D index pairs with S = {2}:
    # an exit cube with an empty image, a fiber that is not acyclic, and an
    # empty fiber.
    script = textwrap.dedent("""
        import CMGDB
        cases = [
            ([1, 2, 3], [1, 3], {1: [], 2: [1, 2, 3], 3: [3]}),
            ([0, 1, 2, 3, 4], [0, 1, 3, 4], {0: [0], 1: [0], 2: [0, 2, 4], 3: [4], 4: [4]}),
            ([1, 2, 3], [1, 3], {1: [1], 2: [], 3: [3]}),
        ]
        for X, A, F in cases:
            print(CMGDB.ComputeConleyIndex(X, A, [5], [False], F, True))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, check=True)
    assert result.stdout == "[]\n[]\n[]\n"
