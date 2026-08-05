import numpy as np
import pytest

import CMGDB


def _point_map(x):
    return [x[i] / (2.0 - x[i]) for i in range(2)]


def _build_cached_graph():
    calls = {"single": 0, "batch": 0}

    def box_map(rect):
        calls["single"] += 1
        return CMGDB.BoxMap(_point_map, rect, padding=False)

    def batch_map(rects):
        calls["batch"] += 1
        return [CMGDB.BoxMap(_point_map, rect, padding=False) for rect in rects]

    model = CMGDB.Model(6, 10, 4, 10000, [0.0, 0.0], [1.2, 1.2], box_map)
    model.set_batch_map(batch_map)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    return morse_graph, map_graph, calls


def _brute_reachable_morse_nodes(map_graph, morse_graph, source):
    owners = {
        int(cell): node
        for node in range(morse_graph.num_vertices())
        for cell in morse_graph.morse_set(node)
    }
    visited = {int(source)}
    stack = [int(source)]
    reached = set()
    while stack:
        vertex = stack.pop()
        if vertex in owners:
            reached.add(owners[vertex])
        for successor in map_graph.adjacencies(vertex):
            successor = int(successor)
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    return reached


def _brute_directed_path_cells(map_graph, morse_graph, source_nodes, target_nodes):
    n_vertices = map_graph.num_vertices()
    sources = {
        int(cell)
        for node in source_nodes
        for cell in morse_graph.morse_set(node)
    }
    targets = {
        int(cell)
        for node in target_nodes
        for cell in morse_graph.morse_set(node)
    }

    forward = set(sources)
    stack = list(sources)
    while stack:
        source = stack.pop()
        for target in map_graph.adjacencies(source):
            target = int(target)
            if target not in forward:
                forward.add(target)
                stack.append(target)

    reverse = [[] for _ in range(n_vertices)]
    for source in range(n_vertices):
        for target in map_graph.adjacencies(source):
            reverse[int(target)].append(source)
    backward = set(targets)
    stack = list(targets)
    while stack:
        target = stack.pop()
        for source in reverse[target]:
            if source not in backward:
                backward.add(source)
                stack.append(source)
    return np.asarray(sorted(forward & backward), dtype=np.uint64)


def test_native_directed_path_cells_matches_forward_intersect_backward():
    morse_graph, map_graph, calls = _build_cached_graph()
    source_nodes = list(range(morse_graph.num_vertices()))
    target_nodes = list(range(morse_graph.num_vertices()))
    calls_before = calls.copy()

    actual = CMGDB.MorseDirectedPathCells(
        map_graph,
        morse_graph,
        source_nodes,
        target_nodes,
    )
    expected = _brute_directed_path_cells(
        map_graph,
        morse_graph,
        source_nodes,
        target_nodes,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint64 and actual.flags.c_contiguous
    assert calls == calls_before


def test_native_reachability_matches_brute_force_all_morse_nodes():
    morse_graph, map_graph, calls = _build_cached_graph()
    queries = list(range(map_graph.num_vertices()))
    calls_before = calls.copy()

    masks = CMGDB.MorseReachabilityMasks(map_graph, morse_graph, queries)
    singleton = CMGDB.MorseSingletonReachability(map_graph, morse_graph, queries)

    expected_masks = []
    expected_singleton = []
    for query in queries:
        reached = _brute_reachable_morse_nodes(map_graph, morse_graph, query)
        expected_masks.append(sum(1 << node for node in reached))
        expected_singleton.append(
            next(iter(reached)) if len(reached) == 1 else -1 if not reached else -2
        )

    np.testing.assert_array_equal(masks, np.asarray(expected_masks, dtype=np.uint64))
    np.testing.assert_array_equal(
        singleton, np.asarray(expected_singleton, dtype=np.int32)
    )
    assert masks.dtype == np.uint64 and masks.flags.c_contiguous
    assert singleton.dtype == np.int32 and singleton.flags.c_contiguous
    # Native reachability only reads the existing CSR; it never evaluates the
    # Python map again.
    assert calls == calls_before


def test_native_singleton_query_preserves_order_and_duplicates():
    morse_graph, map_graph, _ = _build_cached_graph()
    queries = [0, 3, 0, 5]

    result = CMGDB.MorseSingletonReachability(map_graph, morse_graph, queries)

    expected = []
    for query in queries:
        reached = _brute_reachable_morse_nodes(map_graph, morse_graph, query)
        expected.append(next(iter(reached)) if len(reached) == 1 else -1 if not reached else -2)
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.int32))


def test_native_reachability_rejects_out_of_range_query():
    morse_graph, map_graph, _ = _build_cached_graph()

    with pytest.raises(IndexError, match="outside"):
        CMGDB.MorseSingletonReachability(
            map_graph, morse_graph, [map_graph.num_vertices()]
        )


def test_native_reachability_requires_cached_graph(monkeypatch):
    # CMGDB_MAPGRAPH_CACHE=0 is the explicit opt-in to the lazy path, which
    # recomputes adjacencies through the map instead of materializing a CSR.
    # Native reachability must reject such a graph rather than call identity
    # once per adjacency.
    monkeypatch.setenv("CMGDB_MAPGRAPH_CACHE", "0")

    def identity(rect):
        dim = len(rect) // 2
        return list(rect[:dim]) + list(rect[dim:])

    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], identity)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert not map_graph.has_cache()

    with pytest.raises(RuntimeError, match="requires a cached MapGraph"):
        CMGDB.MorseSingletonReachability(map_graph, morse_graph, [0])

    with pytest.raises(RuntimeError, match="requires a cached MapGraph"):
        CMGDB.MorseDirectedPathCells(map_graph, morse_graph, [0], [0])
