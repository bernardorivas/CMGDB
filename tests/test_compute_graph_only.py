"""`Compute*MorseGraphOnly` must equal the paired variant, minus the MapGraph.

`ComputeConleyMorseGraph` and `ComputeMorseGraph` each build a final `MapGraph`
over the whole phase space purely to return it. That is an extra full pass of
the box map -- roughly half of all box-map evaluations -- and callers that do
not consume the MapGraph were paying it for nothing. The `*Only` variants skip
it. These tests pin the resulting Morse graph to be identical either way, and
pin the saving to be real.
"""

from __future__ import annotations

import CMGDB
import pytest


def contraction_box_map(rect):
    """The `tests/test_basic.py` fixture: x -> x / (2 - x), componentwise."""
    dim = len(rect) // 2
    out = []
    for lo in (rect[:dim], rect[dim:]):
        out.append([x / (2.0 - x) for x in lo])
    lower = [min(a, b) for a, b in zip(out[0], out[1])]
    upper = [max(a, b) for a, b in zip(out[0], out[1])]
    return lower + upper


def build_model(counter=None):
    box_map = contraction_box_map
    if counter is not None:

        def counted(rect):
            counter[0] += 1
            return contraction_box_map(rect)

        box_map = counted
    return CMGDB.Model(6, 10, 4, 10000, [0.0, 0.0], [1.2, 1.2], box_map)


def summarize(mg):
    vertices = sorted(mg.vertices())
    return (
        [tuple(str(a) for a in mg.annotations(v)) for v in vertices],
        sorted((int(u), int(w)) for u, w in mg.edges()),
    )


def test_conley_only_matches_paired_variant():
    paired, _map_graph = CMGDB.ComputeConleyMorseGraph(build_model())
    only = CMGDB.ComputeConleyMorseGraphOnly(build_model())
    assert summarize(only) == summarize(paired)
    # Same fixture as tests/test_basic.py, so the answer is known.
    assert only.num_vertices() == 4


def test_morse_only_matches_paired_variant():
    paired, _map_graph = CMGDB.ComputeMorseGraph(build_model())
    only = CMGDB.ComputeMorseGraphOnly(build_model())
    assert summarize(only) == summarize(paired)
    assert only.num_vertices() == 4


def test_conley_only_returns_morse_graph_not_pair():
    result = CMGDB.ComputeConleyMorseGraphOnly(build_model())
    assert not isinstance(result, tuple)
    assert hasattr(result, "vertices")


def test_conley_index_for_cells_matches_node_annotations():
    model = build_model()
    morse_graph = CMGDB.ComputeConleyMorseGraphOnly(model)

    for vertex in morse_graph.vertices():
        recomputed = CMGDB.ComputeConleyIndexForCells(
            model,
            morse_graph,
            morse_graph.morse_set(vertex),
        )
        assert tuple(recomputed) == tuple(morse_graph.annotations(vertex))


@pytest.mark.parametrize(
    "paired_fn,only_fn",
    [
        (CMGDB.ComputeConleyMorseGraph, CMGDB.ComputeConleyMorseGraphOnly),
        (CMGDB.ComputeMorseGraph, CMGDB.ComputeMorseGraphOnly),
    ],
)
def test_only_variant_evaluates_the_map_fewer_times(paired_fn, only_fn):
    paired_calls = [0]
    paired_fn(build_model(paired_calls))
    only_calls = [0]
    only_fn(build_model(only_calls))
    assert only_calls[0] < paired_calls[0], (
        f"expected the Only variant to skip a full box-map pass; "
        f"paired={paired_calls[0]} only={only_calls[0]}"
    )
