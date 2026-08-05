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


def test_batch_cache_has_no_size_ceiling(monkeypatch):
    """No vertex or edge cap: the cache is built for whatever grid it is given.

    Earlier revisions refused, up front, any graph above
    ``CMGDB_MAPGRAPH_MAX_VERTICES`` / ``CMGDB_MAPGRAPH_MAX_EDGES`` whenever a
    batch map was installed -- while the same graph on the scalar path only
    degraded silently. Both env vars are gone; a run that does not fit is left
    to fail where it actually runs out of memory.
    """
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

    for name in (
        "CMGDB_MAPGRAPH_MAX_VERTICES",
        "CMGDB_MAPGRAPH_MAX_EDGES",
    ):
        monkeypatch.delenv(name, raising=False)

    model = CMGDB.Model(4, 4, 4, 10000, [0.0, 0.0], [1.0, 1.0], identity)
    model.set_batch_map(identity_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.num_vertices() == 16
    assert map_graph.has_cache()
    assert calls["batch"] > 0
    assert calls["single"] == 0


def test_former_cap_env_vars_are_inert(monkeypatch):
    """Setting the removed variables must not resurrect the old refusal."""
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_VERTICES", "1")
    monkeypatch.setenv("CMGDB_MAPGRAPH_MAX_EDGES", "1")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() == 16


def test_reserve_edges_is_a_hint_not_a_ceiling(monkeypatch):
    """A reserve smaller than the real edge count grows rather than failing."""
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_EDGES", "2")
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES", "1")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.has_cache()
    assert map_graph.num_cached_edges() == 16


def test_reserve_hints_reject_malformed_values(monkeypatch):
    """A typo in a sizing hint is still an error -- silently ignoring it would
    drop the hint the caller asked for."""
    monkeypatch.setenv("CMGDB_MAPGRAPH_RESERVE_EDGES", "not-a-number")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(ValueError, match="positive base-10 integer"):
        CMGDB.ComputeMorseGraph(model)


def test_cache_can_be_disabled_explicitly(monkeypatch):
    """CMGDB_MAPGRAPH_CACHE=0 selects the lazy, memory-lean path.

    This is the supported way to ask for low memory. It replaces provoking the
    same fallback with an artificially low edge cap, which also refused runs
    that would have fit.
    """
    monkeypatch.setenv("CMGDB_MAPGRAPH_CACHE", "0")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    _, map_graph = CMGDB.ComputeMorseGraph(model)

    assert not map_graph.has_cache()
    assert map_graph.num_cached_edges() == 0


def test_cache_toggle_rejects_unrecognized_values(monkeypatch):
    monkeypatch.setenv("CMGDB_MAPGRAPH_CACHE", "maybe")
    model = CMGDB.Model(2, 2, 2, 10000, [0.0, 0.0], [1.0, 1.0], f)
    model.set_batch_map(f_batch)

    with pytest.raises(ValueError, match="CMGDB_MAPGRAPH_CACHE"):
        CMGDB.ComputeMorseGraph(model)
