"""Regression tests for the MapGraph CSR cache: concurrent build_cache()
calls."""

import threading

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
