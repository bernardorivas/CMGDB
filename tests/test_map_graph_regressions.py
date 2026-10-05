"""Regression tests for the MapGraph CSR cache: concurrent build_cache()
calls, the max_cached_edges limit and explicit cache requests."""

import threading
import warnings

import pytest

import CMGDB


def product_model(subdiv=10):
    """Uniform-grid product map x -> x/(2-x) on [0, 1.2]^2."""
    def f(x):
        return [x[0] / (2.0 - x[0]), x[1] / (2.0 - x[1])]

    return CMGDB.Model(subdiv, subdiv, subdiv, 10000, [0.0, 0.0], [1.2, 1.2],
                       lambda rect: CMGDB.BoxMap(f, rect))


def product_batch(rects):
    return CMGDB.BoxMapBatch(lambda X: X / (2.0 - X), rects)


def all_adjacencies(map_graph):
    return [list(map_graph.adjacencies(v))
            for v in range(map_graph.num_vertices())]


# ---------------------------------------------------------------------------
# build_cache while another build of the same graph is running (C32)
# ---------------------------------------------------------------------------

def test_concurrent_build_cache_raises_and_leaves_a_correct_cache():
    # A batch map runs Python, so another thread can call build_cache() on the
    # same graph while a build waits in the map. That second build used to
    # interleave its rows with the first one's: twice the edges and wrong
    # adjacency lists behind has_cache() == True.
    model = product_model()
    inside = threading.Event()
    proceed = threading.Event()
    armed = [False]

    def batch(rects):
        if armed[0]:
            armed[0] = False
            inside.set()
            proceed.wait(timeout=60)
        return product_batch(rects)

    model.set_batch_map(batch)
    _, map_graph = CMGDB.ComputeMorseGraph(model, batch_chunk_size=64,
                                           cache_map_graph=False)
    reference = all_adjacencies(map_graph)
    errors = []

    def build():
        try:
            map_graph.build_cache()
        except Exception as error:  # reported by the asserts below
            errors.append(error)

    armed[0] = True
    builder = threading.Thread(target=build)
    builder.start()
    try:
        assert inside.wait(timeout=60)
        with pytest.raises(RuntimeError, match="already running"):
            map_graph.build_cache()
    finally:
        proceed.set()
        builder.join(timeout=60)
    assert not errors
    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() == sum(map(len, reference))
    assert all_adjacencies(map_graph) == reference
    map_graph.csr_view()  # validates the offsets and rows


def test_build_cache_called_from_the_map_raises_inside_it():
    model = product_model()
    graphs = []
    errors = []

    def batch(rects):
        if graphs:
            try:
                graphs.pop().build_cache()
            except RuntimeError as error:
                errors.append(error)
        return product_batch(rects)

    model.set_batch_map(batch)
    _, map_graph = CMGDB.ComputeMorseGraph(model, batch_chunk_size=64,
                                           cache_map_graph=False)
    reference = all_adjacencies(map_graph)
    graphs.append(map_graph)
    map_graph.build_cache()
    assert len(errors) == 1 and "already running" in str(errors[0])
    assert map_graph.num_cached_edges() == sum(map(len, reference))
    assert all_adjacencies(map_graph) == reference


# ---------------------------------------------------------------------------
# max_cached_edges is an exact limit, checked before each row (C34)
# ---------------------------------------------------------------------------

def front_loaded_model(depth=14):
    """1D map on [0, 1] whose boxes left of 1/16 have images 400 times wider
    than elsewhere, so the tree-order sweep meets the dense part first."""
    def F(rect):
        a, b = rect
        h = b - a
        lo = a + 0.25 * h
        width = 100.0 * h if a < 1.0 / 16 else 0.25 * h
        return [lo, min(1.0 - 0.1 * h, lo + width)]

    return CMGDB.Model(depth, depth, depth, 10000, [0.0], [1.0], F)


def test_max_cached_edges_keeps_a_cache_that_fits():
    # The cache used to be abandoned once the edge count projected from the
    # chunks seen so far passed twice the limit. This graph is dense early in
    # the sweep, so even a limit of five times its edge count dropped it.
    _, full = CMGDB.ComputeMorseGraph(front_loaded_model(),
                                      batch_chunk_size=1024)
    edges = full.num_cached_edges()
    assert full.has_cache() and edges > 0
    for limit in (edges, 3 * edges // 2):
        _, capped = CMGDB.ComputeMorseGraph(front_loaded_model(),
                                            batch_chunk_size=1024,
                                            max_cached_edges=limit)
        assert capped.has_cache(), limit
        assert capped.num_cached_edges() == edges
    assert all_adjacencies(capped) == all_adjacencies(full)
    _, over = CMGDB.ComputeMorseGraph(front_loaded_model(),
                                      batch_chunk_size=1024,
                                      max_cached_edges=edges - 1)
    assert not over.has_cache()


def test_max_cached_edges_stops_a_whole_grid_chunk_at_the_limit():
    # With batch_chunk_size=0 (one chunk for the whole grid) the limit was
    # checked only after every row had been stored, so it bounded nothing.
    # The rows of the returned map_graph's abandoned build are counted here
    # as scalar map calls: every row of this map has at least one edge.
    calls = [0]

    def f(x):
        return [x[0] / (2.0 - x[0]), x[1] / (2.0 - x[1])]

    def F(rect):
        calls[0] += 1
        return CMGDB.BoxMap(f, rect)

    model = CMGDB.Model(10, 10, 10, 10000, [0.0, 0.0], [1.2, 1.2], F)

    def run(**kwargs):
        calls[0] = 0
        _, map_graph = CMGDB.ComputeMorseGraph(
            model, cache_transition_graph=False, batch_chunk_size=0,
            max_cached_edges=10, **kwargs)
        return calls[0], map_graph

    lazy_calls, _ = run(cache_map_graph=False)
    eager_calls, map_graph = run()
    assert map_graph.num_vertices() == 1024
    assert not map_graph.has_cache()
    assert eager_calls - lazy_calls <= 11


# ---------------------------------------------------------------------------
# Explicit cache requests under max_cached_edges (C35)
# ---------------------------------------------------------------------------

def test_build_cache_is_not_bound_by_the_limit_of_the_computation():
    # The returned map_graph kept the call's max_cached_edges, so
    # build_cache(), the documented upgrade, spent a full map pass, dropped
    # the result and left the graph lazy, with no way to lift the limit.
    _, reference = CMGDB.ComputeMorseGraph(product_model(subdiv=8),
                                           cache_map_graph=True)
    edges = reference.num_cached_edges()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(
        product_model(subdiv=8), max_cached_edges=edges - 1,
        cache_map_graph=False)
    map_graph.build_cache()
    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() == edges
    assert all_adjacencies(map_graph) == all_adjacencies(reference)
    CMGDB.MorseReachabilityMasks(map_graph, morse_graph, [0])


def test_lazy_graph_errors_name_build_cache_past_the_limit(tmp_path):
    # The csr_view() and checkpoint errors said that a graph over
    # max_cached_edges stays lazy, which pointed away from build_cache(),
    # the remedy that the limit does not bound.
    _, reference = CMGDB.ComputeMorseGraph(product_model(subdiv=8),
                                           cache_map_graph=True)
    edges = reference.num_cached_edges()
    _, map_graph = CMGDB.ComputeMorseGraph(product_model(subdiv=8),
                                           max_cached_edges=edges - 1)
    assert not map_graph.has_cache()
    remedy = r"build_cache\(\), which that limit does not bound"
    with pytest.raises(RuntimeError, match=remedy):
        map_graph.csr_view()
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=10**6, max_edges=10**6, max_payload_bytes=10**8)
    with pytest.raises(RuntimeError, match=remedy):
        CMGDB.write_map_graph_csr_checkpoint(
            map_graph, tmp_path / "lazy.csr",
            configuration={"model": "product"}, caps=caps)
    map_graph.build_cache()
    _, targets = map_graph.csr_view()
    assert len(targets) == edges


def test_build_cache_raises_over_its_own_limit():
    _, map_graph = CMGDB.ComputeMorseGraph(product_model(subdiv=8),
                                           cache_map_graph=False)
    lazy = all_adjacencies(map_graph)
    edges = sum(map(len, lazy))
    with pytest.raises(RuntimeError, match=f"max_cached_edges={edges - 1} "):
        map_graph.build_cache(max_cached_edges=edges - 1)
    assert not map_graph.has_cache()
    assert all_adjacencies(map_graph) == lazy
    map_graph.build_cache(max_cached_edges=edges)
    assert map_graph.num_cached_edges() == edges


def test_explicit_cache_map_graph_warns_when_the_limit_drops_it():
    model = product_model(subdiv=8)
    _, reference = CMGDB.ComputeMorseGraph(model, cache_map_graph=True)
    edges = reference.num_cached_edges()
    for compute in (CMGDB.ComputeMorseGraph, CMGDB.ComputeConleyMorseGraph):
        with pytest.warns(RuntimeWarning,
                          match=f"max_cached_edges={edges - 1} .*build_cache"):
            _, map_graph = compute(model, cache_map_graph=True,
                                   max_cached_edges=edges - 1)
        assert not map_graph.has_cache()
        map_graph.build_cache()
        assert map_graph.num_cached_edges() == edges
    # No warning within the limit, nor when the cache was not asked for.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, map_graph = CMGDB.ComputeMorseGraph(model, cache_map_graph=True,
                                               max_cached_edges=edges)
        assert map_graph.has_cache()
        _, map_graph = CMGDB.ComputeMorseGraph(model,
                                               max_cached_edges=edges - 1)
        assert not map_graph.has_cache()
