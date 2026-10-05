"""Regression tests for CMGDB.morse_lattice.

A node of a Conley Morse graph whose index CMGDB left undefined has no
annotation in the PlotMorseGraph output. The lattice helpers that need every
index raised "has no Conley-index annotation ... requires a Conley Morse
graph" for it, although the graph is a Conley Morse graph.
"""

import tempfile
from pathlib import Path

import pytest

import CMGDB
from CMGDB.morse_graph_parser import MorseGraph
from CMGDB.morse_lattice import attractor_type, nontrivial_cmgraph


def _mg(text: str) -> MorseGraph:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mg.dot"
        p.write_text(text)
        return MorseGraph.from_dot(p)


# Node 1's index is undefined; the other nodes are annotated.
UNDEFINED_NODE_DOT = """digraph G {
0 [label="0 : (0, 0, x-1)"];
1 [label="1"];
2 [label="2 : (x-1, 0, 0)"];
0 -> 1;
1 -> 2;
}
"""

# No node is annotated.
UNANNOTATED_DOT = """digraph G {
0 [label="0"];
1 [label="1"];
0 -> 1;
}
"""


def test_undefined_index_is_named_as_such():
    with pytest.raises(ValueError, match="Morse node 1 .* has an undefined Conley index"):
        nontrivial_cmgraph(_mg(UNDEFINED_NODE_DOT))
    with pytest.raises(ValueError, match="undefined Conley index"):
        attractor_type(_mg(UNDEFINED_NODE_DOT), frozenset({1, 2}))


def test_graph_without_annotations_names_both_causes():
    with pytest.raises(ValueError, match="computed without Conley indices, or none of them is defined"):
        nontrivial_cmgraph(_mg(UNANNOTATED_DOT))


def test_undefined_index_from_a_computed_graph():
    # The saddle at the origin of (10 x, y / 2) on [-1, 1] x [0, 1] lies on
    # the boundary of the phase space, and its index is undefined; the other
    # Morse set has a trivial index.
    f = lambda x: [10.0 * x[0], 0.5 * x[1]]
    model = CMGDB.Model(6, 6, 6, 10000, [-1.0, 0.0], [1.0, 1.0],
                        lambda rect: CMGDB.BoxMap(f, rect))
    morse_graph = CMGDB.ComputeConleyMorseGraphOnly(model)
    annotations = [morse_graph.annotations(v) for v in range(morse_graph.num_vertices())]
    assert [] in annotations and ["0", "0", "0"] in annotations
    parsed = _mg(CMGDB.PlotMorseGraph(morse_graph).source)
    with pytest.raises(ValueError, match="has an undefined Conley index"):
        nontrivial_cmgraph(parsed)
