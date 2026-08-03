import CMGDB
import pytest


def f(rect):
    dim = len(rect) // 2
    return list(rect[:dim]) + list(rect[dim:])


def f_batch(rects):
    return [f(rect) for rect in rects]


def test_model_accepts_batch_map_callback():
    model = CMGDB.Model(4, 4, 4, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() >= 1


def test_map_graph_cache_uses_batch_map_callback():
    calls = {"batch": 0}

    def identity(rect):
        dim = len(rect) // 2
        return list(rect[:dim]) + list(rect[dim:])

    def identity_batch(rects):
        calls["batch"] += 1
        return [identity(rect) for rect in rects]

    model = CMGDB.Model(4, 4, 4, 10000, [0.0, 0.0], [1.0, 1.0], identity)
    model.set_batch_map(identity_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert calls["batch"] > 0
    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() > 0


def test_batch_cache_vertex_limit_is_inclusive(monkeypatch):
    calls = {"single": 0, "batch": 0}

    def identity(rect):
        calls["single"] += 1
        dim = len(rect) // 2
        return list(rect[:dim]) + list(rect[dim:])

    def identity_batch(rects):
        calls["batch"] += 1
        return [
            list(rect[: len(rect) // 2]) + list(rect[len(rect) // 2 :])
            for rect in rects
        ]

    # A fixed-depth-4 grid has exactly 2^4 = 16 cells. Equality must remain
    # cache-eligible so the default 2^24 limit includes a level-24 grid.
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_VERTICES", "16")
    model = CMGDB.Model(4, 4, 4, 10000, [0.0, 0.0], [1.0, 1.0], identity)
    model.set_batch_map(identity_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.num_vertices() == 16
    assert map_graph.has_cache()
    assert calls["batch"] > 0
    assert calls["single"] == 0


def test_batch_cache_refuses_scalar_fallback_above_vertex_limit(monkeypatch):
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_VERTICES", "15")
    model = CMGDB.Model(4, 4, 4, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(RuntimeError, match="refusing to fall back to per-cell"):
        CMGDB.ComputeMorseGraph(model)


def test_batch_cache_edge_limit_is_configurable_and_inclusive(monkeypatch):
    # Identity on the four-cell fixed-depth-2 grid has exactly 16 cover edges.
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_EDGES", "16")
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_EDGES", "16")
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES", "1")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() == 16


def test_batch_cache_refuses_scalar_fallback_above_edge_limit(monkeypatch):
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_EDGES", "15")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(RuntimeError, match="CMGDB_MAPGRAPH_MAX_EDGES=15"):
        CMGDB.ComputeMorseGraph(model)


def test_batch_cache_rejects_reserve_above_edge_limit(monkeypatch):
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_EDGES", "16")
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_EDGES", "17")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(ValueError, match="RESERVE_EDGES=17"):
        CMGDB.ComputeMorseGraph(model)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CMGDB_MAPGRAPH_MAX_VERTICES", "0"),
        ("CMGDB_MAPGRAPH_MAX_EDGES", "not-a-number"),
    ],
)
def test_batch_cache_rejects_invalid_limits(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(ValueError, match="positive base-10 integer"):
        CMGDB.ComputeMorseGraph(model)
