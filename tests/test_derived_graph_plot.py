"""Tests for graphviz rendering of DerivedGraph lattices and nontrivial graphs."""

import re
import tempfile
import warnings
from pathlib import Path

import matplotlib
import pytest

from CMGDB.morse_graph_parser import MorseGraph
from CMGDB.morse_lattice import DerivedGraph, lattice_of_attractors, lattice_of_repellers, nontrivial_cmgraph
from CMGDB.derived_graph_plot import _DEFAULT_CLIST, plot_derived_graph
from CMGDB.PlotMorseGraph import PlotMorseGraph


def _mg(text: str) -> MorseGraph:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mg.dot"
        p.write_text(text)
        return MorseGraph.from_dot(p)


class _StubMorseGraph:
    """Minimal object satisfying PlotMorseGraph's vertices()/edges() API."""

    def __init__(self, morse_graph: MorseGraph):
        self._nodes = list(morse_graph.nodes)
        self._edges = [(u, v) for u in morse_graph.nodes for v in morse_graph.edges.get(u, [])]

    def vertices(self):
        return self._nodes

    def edges(self):
        return self._edges


# 3-node DAG, 0 -> 1, 0 -> 2; minimal (attractor) nodes {1, 2}.
TWO_SINK_DOT = """digraph G {
0 [label="0"];
1 [label="1"];
2 [label="2"];
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

# A chain 0 -> 1 beside an isolated node 2, all nontrivial.
ISOLATED_NODE_DOT = """digraph G {
0 [label="0 : (0, x - 1, 0)"];
1 [label="1 : (x - 1, 0, 0)"];
2 [label="2 : (x - 1, 0, 0)"];
0 -> 1;
}
"""


def _fillcolors(source) -> dict[int, str]:
    return {
        int(v): color
        for v, color in re.findall(r'^(\d+) \[.*fillcolor="([^"]+)"', source.source, re.M)
    }


def _hex(index: int) -> str:
    return matplotlib.colors.to_hex(_DEFAULT_CLIST[index], keep_alpha=True)


def test_lattice_colors_match_morse_nodes():
    # Lattice of TWO_SINK: {} < {1}, {2} < {1,2} < {0,1,2}.
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    colors = _fillcolors(plot_derived_graph(lattice))
    by_set = {lattice.sets[v]: v for v in lattice.nodes}
    assert colors[by_set[frozenset()]] == "#cdcdcd"
    assert colors[by_set[frozenset({1})]] == _hex(1)
    assert colors[by_set[frozenset({2})]] == _hex(2)
    # Joins take colors beyond the Morse-node range, in vertex order.
    assert colors[by_set[frozenset({1, 2})]] == _hex(3)
    assert colors[by_set[frozenset({0, 1, 2})]] == _hex(4)


def test_lattice_singletons_match_plot_morse_graph_output():
    morse_graph = _mg(TWO_SINK_DOT)
    mg_colors = _fillcolors(PlotMorseGraph(_StubMorseGraph(morse_graph)))
    lattice = lattice_of_attractors(morse_graph)
    lat_colors = _fillcolors(plot_derived_graph(lattice))
    by_set = {lattice.sets[v]: v for v in lattice.nodes}
    for m in (1, 2):
        assert lat_colors[by_set[frozenset({m})]] == mg_colors[m]


def test_repeller_lattice_singletons_match_plot_morse_graph_output():
    morse_graph = _mg(TWO_SINK_DOT)
    mg_colors = _fillcolors(PlotMorseGraph(_StubMorseGraph(morse_graph)))
    lattice = lattice_of_repellers(morse_graph)
    lat_colors = _fillcolors(plot_derived_graph(lattice))
    by_set = {lattice.sets[v]: v for v in lattice.nodes}
    # Sole singleton upset: {0}, the maximal node.
    assert lat_colors[by_set[frozenset({0})]] == mg_colors[0]


def test_nontrivial_graph_colors_match_plot_morse_graph_output():
    # Node 1 is pruned; survivors 0 and 2 keep their PlotMorseGraph colors.
    morse_graph = _mg(TRIVIAL_CHAIN_DOT)
    mg_colors = _fillcolors(PlotMorseGraph(_StubMorseGraph(morse_graph)))
    graph = nontrivial_cmgraph(morse_graph)
    colors = _fillcolors(plot_derived_graph(graph))
    assert set(colors) == {0, 2}
    assert colors[0] == mg_colors[0]
    assert colors[2] == mg_colors[2]
    source = plot_derived_graph(graph).source
    assert 'label="0 : (x - 1, 0, 0)"' in source
    assert "0 -> 2;" in source


def test_lattice_labels_and_edges_rendered():
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    source = plot_derived_graph(lattice).source
    assert 'label="0 : \\{ \\}"' in source
    for u in lattice.nodes:
        for v in lattice.edges.get(u, []):
            assert f"{u} -> {v};" in source


def test_small_clist_cycles_with_warning():
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    with pytest.warns(UserWarning, match="colors will repeat"):
        source = plot_derived_graph(lattice, clist=["#ff0000", "#00ff00"])
    colors = _fillcolors(source)
    by_set = {lattice.sets[v]: v for v in lattice.nodes}
    # Index 2 wraps to color 0, index 3 to color 1.
    assert colors[by_set[frozenset({2})]] == matplotlib.colors.to_hex("#ff0000", keep_alpha=True)
    assert colors[by_set[frozenset({1, 2})]] == matplotlib.colors.to_hex("#00ff00", keep_alpha=True)


def test_default_palette_does_not_warn_when_cycling():
    # 41 nodes force color indices past the 40-color default palette; only
    # user-supplied palettes should warn on wrap.
    lines = [f'{i} [label="{i}"];' for i in range(41)]
    mg = _mg("digraph G {\n" + "\n".join(lines) + "\n}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_derived_graph(mg)


def test_large_listed_cmap_spreads_indices():
    # viridis is a 256-color ListedColormap; indices must spread across it,
    # not int-index the first few near-identical dark entries.
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    colors = _fillcolors(plot_derived_graph(lattice, cmap=matplotlib.cm.viridis))
    colored = [c for c in colors.values() if c != "#cdcdcd"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    # Indices in use are 1, 2 (singletons {1}, {2}) and 3, 4 (joins).
    expected = {
        matplotlib.colors.to_hex(matplotlib.cm.viridis(norm(k)), keep_alpha=True)
        for k in (1, 2, 3, 4)
    }
    assert set(colored) == expected


def test_continuous_cmap_spreads_indices():
    # jet is a LinearSegmentedColormap (no .colors): the Normalize branch.
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    colors = _fillcolors(plot_derived_graph(lattice, cmap=matplotlib.cm.jet))
    colored = [c for c in colors.values() if c != "#cdcdcd"]
    assert len(set(colored)) == len(colored)
    assert matplotlib.colors.to_hex(matplotlib.cm.jet(1.0), keep_alpha=True) in colored


def test_isolated_node_does_not_flatten_hierarchy():
    # Node 2 is both sink and source; it must not appear in the source rank
    # row, or dot merges the two ranks and flattens the chain 0 -> 1.
    graph = nontrivial_cmgraph(_mg(ISOLATED_NODE_DOT))
    assert set(graph.nodes) == {0, 1, 2}
    plain = plot_derived_graph(graph).pipe(format="plain").decode()
    ys = {
        parts[1]: float(parts[3])
        for parts in (line.split() for line in plain.splitlines())
        if parts and parts[0] == "node"
    }
    assert ys["0"] != ys["1"]
    # The isolated node sits on the sink row with the chain's attractor.
    assert ys["2"] == ys["1"]


def test_record_shape_renders():
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    svg = plot_derived_graph(lattice, shape="record").pipe(format="svg").decode()
    assert "{1, 2}" in svg


def test_quote_and_backslash_labels_escaped():
    graph = DerivedGraph(nodes=[0, 1], edges={0: [1]}, labels={0: 'say "hi"', 1: "end\\"}, sets=None)
    svg = plot_derived_graph(graph).pipe(format="svg").decode()
    assert "say" in svg


def test_none_shape_and_margin_use_defaults():
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    source = plot_derived_graph(lattice, shape=None, margin=None).source
    assert "shape=ellipse" in source
    assert 'margin="0.11, 0.055"' in source


def test_parsed_morse_graph_input():
    # A parsed MorseGraph plots directly: colored by node id, labels not double-prefixed.
    mg = _mg(TRIVIAL_CHAIN_DOT)
    source = plot_derived_graph(mg)
    colors = _fillcolors(source)
    assert colors == {0: _hex(0), 1: _hex(1), 2: _hex(2)}
    assert 'label="0 : (x - 1, 0, 0)"' in source.source
    assert "0 : 0" not in source.source


def test_empty_clist_falls_back_to_default_palette():
    lattice = lattice_of_attractors(_mg(TWO_SINK_DOT))
    assert _fillcolors(plot_derived_graph(lattice, clist=[])) == _fillcolors(plot_derived_graph(lattice))
