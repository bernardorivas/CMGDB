"""Native reachability queries on hierarchical runs (C24, C30, C31).

On a hierarchical run (phase_subdiv_init < phase_subdiv_min) the Morse graph
comes from grids coarser than the returned map_graph, and corner-sampled box
maps are not monotone under refinement. The Morse graph's edges then need not
match reachability between the cells of map_graph, in either direction, and
map_graph can have cycles outside the Morse sets. MorseReachabilityMasks,
MorseSingletonReachability and MorseDirectedPathCells are defined on the
cells of map_graph; these tests compare them with brute-force BFS oracles on
such runs. Each test also checks that its run has the property it is about.
"""

import math

import numpy as np
import pytest

import CMGDB


def _product_f(x):
    return [x[d] / (2.0 - x[d]) for d in range(len(x))]


def _product_box_map(rect):
    return CMGDB.BoxMap(_product_f, rect)


def product_model(subdiv_min, subdiv_max, subdiv_init=None):
    """x -> x/(2-x) on [0, 1.2]^2; init 0 with the 5-argument constructor."""
    if subdiv_init is None:
        return CMGDB.Model(subdiv_min, subdiv_max,
                           [0.0, 0.0], [1.2, 1.2], _product_box_map)
    return CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, 10000,
                       [0.0, 0.0], [1.2, 1.2], _product_box_map)


def ricker_model():
    """Ricker map x exp(3(1-x)) on [0, 4], Model(6, 6, [0], [4], F)."""
    def f(x):
        return [x[0] * math.exp(3.0 * (1.0 - x[0]))]

    return CMGDB.Model(6, 6, [0.0], [4.0], lambda rect: CMGDB.BoxMap(f, rect))


def leslie_model(subdiv_min=12, subdiv_max=14):
    """The Leslie model of tests/test_map_graph_cache.py (init 0)."""
    theta = [19.6, 23.68]

    def f(x):
        s = x[0] + x[1]
        return [(theta[0] * x[0] + theta[1] * x[1]) * math.exp(-0.1 * s),
                0.7 * x[0]]

    return CMGDB.Model(subdiv_min, subdiv_max, [-0.001, -0.001], [90.0, 70.0],
                       lambda rect: CMGDB.BoxMap(f, rect))


class CellGraph:
    """The cached map_graph as Python lists, with brute-force oracles."""

    def __init__(self, model):
        self.morse_graph, self.map_graph = CMGDB.ComputeMorseGraph(
            model, cache_map_graph=True)
        assert self.map_graph.has_cache()
        assert model.phase_subdiv_init() < model.phase_subdiv_min()
        self.n = self.map_graph.num_vertices()
        self.adjacency = [[int(w) for w in self.map_graph.adjacencies(v)]
                          for v in range(self.n)]
        self.reverse = [[] for _ in range(self.n)]
        for v, targets in enumerate(self.adjacency):
            for w in targets:
                self.reverse[w].append(v)
        self.nodes = self.morse_graph.num_vertices()
        self.morse_sets = [set(int(c) for c in self.morse_graph.morse_set(k))
                           for k in range(self.nodes)]

    @staticmethod
    def bfs(adjacency, seeds):
        seen = set(seeds)
        stack = list(seeds)
        while stack:
            v = stack.pop()
            for w in adjacency[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    def reached_nodes(self, cell):
        reach = self.bfs(self.adjacency, [cell])
        return {k for k, cells in enumerate(self.morse_sets) if reach & cells}

    def path_cells(self, sources, targets):
        forward = self.bfs(self.adjacency,
                           sorted(set().union(*(self.morse_sets[s] for s in sources))))
        backward = self.bfs(self.reverse,
                            sorted(set().union(*(self.morse_sets[t] for t in targets))))
        return sorted(forward & backward)

    def cell_level_pairs(self):
        """Ordered pairs of distinct Morse nodes joined by a cell path."""
        pairs = set()
        for a in range(self.nodes):
            reach = self.bfs(self.adjacency, sorted(self.morse_sets[a]))
            pairs.update((a, b) for b in range(self.nodes)
                         if b != a and reach & self.morse_sets[b])
        return pairs

    def morse_closure_pairs(self):
        """Ordered pairs of distinct Morse nodes in the Morse graph's closure."""
        edges = {(a, b) for a, b in self.morse_graph.edges_unreduced() if a != b}
        closure = {k: {k} for k in range(self.nodes)}
        changed = True
        while changed:
            changed = False
            for a, b in edges:
                if not closure[b] <= closure[a]:
                    closure[a] |= closure[b]
                    changed = True
        return {(a, b) for a in range(self.nodes) for b in closure[a] if b != a}

    def has_cycle_outside_morse_sets(self):
        """Whether the cells outside the Morse sets carry a directed cycle."""
        morse_cells = set().union(*self.morse_sets)
        outside = [v for v in range(self.n) if v not in morse_cells]
        indegree = {v: 0 for v in outside}
        for v in outside:
            for w in self.adjacency[v]:
                if w in indegree:
                    indegree[w] += 1
        ready = [v for v in outside if indegree[v] == 0]
        removed = 0
        while ready:
            v = ready.pop()
            removed += 1
            for w in self.adjacency[v]:
                if w in indegree:
                    indegree[w] -= 1
                    if indegree[w] == 0:
                        ready.append(w)
        return removed < len(outside)

    def assert_queries_match_oracles(self):
        queries = list(range(self.n))
        expected = [self.reached_nodes(q) for q in queries]

        masks = CMGDB.MorseReachabilityMasks(
            self.map_graph, self.morse_graph, queries)
        np.testing.assert_array_equal(
            masks,
            np.asarray([sum(1 << k for k in nodes) for nodes in expected],
                       dtype=np.uint64))
        assert masks.dtype == np.uint64

        summary = CMGDB.MorseSingletonReachability(
            self.map_graph, self.morse_graph, queries)
        np.testing.assert_array_equal(
            summary,
            np.asarray([next(iter(nodes)) if len(nodes) == 1
                        else -1 if not nodes else -2 for nodes in expected],
                       dtype=np.int32))
        assert summary.dtype == np.int32

        everything = list(range(self.nodes))
        pairs = [([s], [t]) for s in everything for t in everything]
        pairs.append((everything, everything))
        for sources, targets in pairs:
            cells = CMGDB.MorseDirectedPathCells(
                self.map_graph, self.morse_graph, sources, targets)
            assert cells.dtype == np.uint64
            assert [int(c) for c in cells] == self.path_cells(sources, targets), (
                f"path cells {sources} -> {targets}")


@pytest.mark.parametrize("model_args", [(6, 10, 0), (8, 8, 0), (6, 6)])
def test_queries_ignore_coarse_morse_edges(model_args):
    # C24, C31: the Morse graph carries coarse-level edges with no cell path
    # in map_graph, which the queries used to report as reachable.
    graph = CellGraph(product_model(*model_args))
    assert graph.morse_closure_pairs() - graph.cell_level_pairs()
    graph.assert_queries_match_oracles()


def test_product_6_10_morse_edge_without_cell_path_has_no_path_cells():
    # C24: Model(6, 10, 0, ...) has a Morse edge between two nodes that no
    # cell path joins; the cells on a path between them form an empty set.
    graph = CellGraph(product_model(6, 10, 0))
    spurious = sorted(graph.morse_closure_pairs() - graph.cell_level_pairs())
    assert spurious
    for source, target in spurious:
        cells = CMGDB.MorseDirectedPathCells(
            graph.map_graph, graph.morse_graph, [source], [target])
        assert cells.size == 0


def test_queries_follow_cell_paths_missing_from_the_morse_graph():
    # C31: in map_graph both Morse sets of the Ricker run lie in one strongly
    # connected component, through a coarse cell outside them, although the
    # Morse graph has a single edge. Cells of the lower set used to be
    # reported as single-node basins.
    graph = CellGraph(ricker_model())
    assert graph.cell_level_pairs() - graph.morse_closure_pairs()
    graph.assert_queries_match_oracles()


@pytest.mark.parametrize("model_args", [(8, 8), (12, 14)])
def test_queries_handle_cycles_outside_the_morse_sets(model_args):
    # C30: all three queries raised "found a directed cycle not covered by
    # the supplied Morse sets" on the Leslie model's hierarchical runs.
    graph = CellGraph(leslie_model(*model_args))
    assert graph.has_cycle_outside_morse_sets()
    graph.assert_queries_match_oracles()


def test_queries_in_any_order_reuse_earlier_answers():
    # The queries keep what earlier query cells reached; asking in another
    # order, with repeats, must give the same answers.
    graph = CellGraph(leslie_model(8, 8))
    queries = list(range(graph.n))[::-1] + [0, graph.n - 1, 0]
    expected = [graph.reached_nodes(q) for q in queries]
    masks = CMGDB.MorseReachabilityMasks(graph.map_graph, graph.morse_graph, queries)
    assert [int(m) for m in masks] == [sum(1 << k for k in nodes) for nodes in expected]
    summary = CMGDB.MorseSingletonReachability(
        graph.map_graph, graph.morse_graph, queries)
    assert [int(s) for s in summary] == [
        next(iter(nodes)) if len(nodes) == 1 else -1 if not nodes else -2
        for nodes in expected]
