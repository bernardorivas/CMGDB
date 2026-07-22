"""Tests for lattice-of-attractors and nontrivial-Conley-graph derivations.

These operate natively on the pure-Python ``MorseGraph`` (no DSGRN/pychomp).
"""

import tempfile
from pathlib import Path

from CMGDB.morse_graph_parser import MorseGraph
from CMGDB.morse_lattice import (
    attractor_type,
    lattice_of_attractors,
    lattice_of_repellers,
    nontrivial_cmgraph,
    transitive_closure,
    transitive_reduction,
)


def _mg(text: str) -> MorseGraph:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mg.dot"
        p.write_text(text)
        return MorseGraph.from_dot(p)


# 3-node DAG, 0 -> 1, 0 -> 2; minimal (attractor) nodes {1, 2}.
TWO_SINK_DOT = """digraph G {
0 [label="0"];
1 [label="1"];
2 [label="2"];
0 -> 1;
0 -> 2;
}
"""

# Two incomparable nontrivial attractors under a nontrivial source.
TWO_NT_SINK_DOT = """digraph G {
0 [label="0 : (x - 1, x, 0)"];
1 [label="1 : (x - 1, 0, 0)"];
2 [label="2 : (x - 1, 0, 0)"];
0 -> 1;
0 -> 2;
}
"""

# Chain with a trivial middle node: 0 (nontrivial) -> 1 (trivial) -> 2 (nontrivial).
TRIVIAL_CHAIN_DOT = """digraph G {
0 [label="0 : (x - 1, 0, 0)"];
1 [label="1 : (0, 0, 0)"];
2 [label="2 : (x - 1, 0, 0)"];
0 -> 1;
1 -> 2;
}
"""


def test_transitive_closure_includes_all_strict_descendants():
    closure = transitive_closure([0, 1, 2], {0: [1, 2], 1: [2]})
    assert closure[0] == {1, 2}
    assert closure[1] == {2}
    assert closure[2] == set()


def test_transitive_reduction_drops_redundant_edge():
    # 0->1, 0->2, 1->2: the direct 0->2 is redundant via 0->1->2.
    reduced = transitive_reduction([0, 1, 2], {0: [1, 2], 1: [2]})
    assert reduced[0] == {1}
    assert reduced[1] == {2}


def test_lattice_of_attractors_two_sinks():
    latt = lattice_of_attractors(_mg(TWO_SINK_DOT))
    assert set(latt.sets.values()) == {
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        frozenset({1, 2}),
        frozenset({0, 1, 2}),
    }
    by_set = {v: k for k, v in latt.sets.items()}
    full = by_set[frozenset({0, 1, 2})]
    pair = by_set[frozenset({1, 2})]
    # Hasse cover: the full attractor covers {1, 2} directly.
    assert pair in latt.edges.get(full, [])


def test_lattice_of_repellers_two_sinks():
    rep = lattice_of_repellers(_mg(TWO_SINK_DOT))
    sets = set(rep.sets.values())
    # Repellers are upsets of the transposed graph: {0} is the elementary repeller.
    assert frozenset() in sets
    assert frozenset({0}) in sets
    assert frozenset({0, 1, 2}) in sets


def test_nontrivial_cmgraph_prunes_trivial_and_keeps_induced_edge():
    nt = nontrivial_cmgraph(_mg(TRIVIAL_CHAIN_DOT))
    assert nt.nodes == [0, 2]
    assert 1 not in nt.nodes
    # 2 is reachable from 0 only through the pruned trivial node 1.
    assert nt.edges.get(0) == [2]


def test_attractor_type_single_nontrivial_node_is_type_2():
    assert attractor_type(_mg(TWO_NT_SINK_DOT), frozenset({1})) == 2


def test_attractor_type_trivial_maximal_is_type_0():
    # {1, 2} in the chain: maximal node is 1 (1 -> 2), which is trivial.
    assert attractor_type(_mg(TRIVIAL_CHAIN_DOT), frozenset({1, 2})) == 0


def test_attractor_type_multiple_nontrivial_maximal_is_type_1():
    # {1, 2}: two incomparable nontrivial maximal nodes.
    assert attractor_type(_mg(TWO_NT_SINK_DOT), frozenset({1, 2})) == 1
