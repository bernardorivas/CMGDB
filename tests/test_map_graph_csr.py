from __future__ import annotations

import gc
import importlib
import json
import weakref
from pathlib import Path

import CMGDB
import numpy as np
import pytest


def _atlas_map_graph():
    model = CMGDB.AtlasModel(0)
    model.add_chart(10, [0.0], [1.0])
    model.add_chart(20, [-1.0], [1.0])
    model.set_active_subgrid(
        [
            (10, 2, [0]),
            (10, 2, [3]),
            (20, 2, [1]),
        ]
    )

    def box_map(chart_id, bounds):
        if chart_id == 20:
            return [(10, [0.9, 0.9])]
        if bounds[0] < 0.5:
            return [(10, [0.1, 0.1]), (10, [0.9, 0.9])]
        return []

    model.set_map(box_map)
    _morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    return map_graph


def _configuration() -> dict[str, object]:
    return {
        "model": "unit-test-atlas",
        "depth": 8,
        "parameters": {"tau": 0.5, "open_exit": True},
    }


def _caps() -> CMGDB.MapGraphCSRCheckpointCaps:
    return CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=100,
        max_edges=100,
        max_payload_bytes=10_000,
    )


def test_native_csr_view_is_zero_copy_read_only_and_exact():
    graph = _atlas_map_graph()
    expected = [graph.adjacencies(index) for index in range(graph.num_vertices())]

    offsets, targets = graph.csr_view()
    assert offsets.dtype == np.int64
    assert targets.dtype == np.int64
    assert not offsets.flags.writeable
    assert not targets.flags.writeable
    assert offsets.tolist() == [0, 2, 2, 3]
    assert targets.tolist() == [0, 1, 1]
    assert [targets[offsets[i] : offsets[i + 1]].tolist() for i in range(3)] == expected

    # The NumPy base owns the native holder, so the view remains valid after
    # the original Python name is gone.
    del graph
    gc.collect()
    assert offsets.tolist() == [0, 2, 2, 3]
    with pytest.raises(ValueError):
        targets[0] = 2


def test_empty_native_graph_has_a_well_formed_csr_view():
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    model.set_active_subgrid([])
    model.set_map(lambda _chart, _bounds: [])
    _morse_graph, graph = CMGDB.ComputeMorseGraph(model)

    offsets, targets = graph.csr_view()
    assert offsets.tolist() == [0]
    assert targets.shape == (0,)
    assert not offsets.flags.writeable
    assert not targets.flags.writeable


def test_atomic_checkpoint_roundtrip_is_mmap_backed_and_deterministic(tmp_path):
    graph = _atlas_map_graph()
    first_path = tmp_path / "first.csr"
    second_path = tmp_path / "second.csr"
    for path in (first_path, second_path):
        CMGDB.write_map_graph_csr_checkpoint(
            graph,
            path,
            configuration=_configuration(),
            caps=_caps(),
        )

    first = CMGDB.load_map_graph_csr_checkpoint(
        first_path,
        expected_configuration=_configuration(),
        caps=_caps(),
    )
    assert isinstance(first.offsets, np.memmap)
    assert isinstance(first.targets, np.memmap)
    assert first.offsets.dtype == np.int64
    assert first.targets.dtype == np.int32
    assert first.num_vertices() == 3
    assert first.num_cached_edges() == 3
    assert [list(first.adjacencies(index)) for index in range(3)] == [
        [0, 1],
        [],
        [1],
    ]
    assert 1 in first[0]
    assert 2 not in first[0]
    assert first[0].as_array().tolist() == [0, 1]

    second = CMGDB.load_map_graph_csr_checkpoint(
        second_path,
        expected_configuration=_configuration(),
        caps=_caps(),
    )
    assert first.fingerprint == second.fingerprint
    assert (first_path / "offsets.npy").read_bytes() == (
        second_path / "offsets.npy"
    ).read_bytes()
    assert (first_path / "targets.npy").read_bytes() == (
        second_path / "targets.npy"
    ).read_bytes()
    first_metadata = json.loads((first_path / "metadata.json").read_text())
    second_metadata = json.loads((second_path / "metadata.json").read_text())
    assert first_metadata == second_metadata


def test_mmap_checkpoint_does_not_retain_native_map_graph(tmp_path):
    graph = _atlas_map_graph()
    native_owner = weakref.ref(graph)
    path = tmp_path / "owner-release.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        path,
        configuration=_configuration(),
        caps=_caps(),
    )
    relation = CMGDB.load_map_graph_csr_checkpoint(
        path,
        expected_configuration=_configuration(),
        caps=_caps(),
    )

    del graph
    gc.collect()
    assert native_owner() is None
    assert [list(relation[index]) for index in relation] == [[0, 1], [], [1]]


def test_forced_int64_targets_and_expected_fingerprint(tmp_path):
    graph = _atlas_map_graph()
    path = tmp_path / "int64.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        path,
        configuration=_configuration(),
        caps=_caps(),
        target_dtype="int64",
    )
    metadata = CMGDB.read_map_graph_csr_metadata(path)
    fingerprint = metadata["fingerprint"]["sha256"]
    loaded = CMGDB.load_map_graph_csr_checkpoint(
        path,
        expected_configuration=_configuration(),
        expected_fingerprint=fingerprint,
        caps=_caps(),
    )
    assert loaded.targets.dtype == np.int64
    with pytest.raises(ValueError, match="fingerprint differs"):
        CMGDB.load_map_graph_csr_checkpoint(
            path,
            expected_configuration=_configuration(),
            expected_fingerprint="0" * 64,
            caps=_caps(),
        )


def test_caps_are_checked_before_requesting_native_view(tmp_path):
    class ViewMustNotRun:
        def __init__(self):
            self.called = False

        def has_cache(self):
            return True

        def num_vertices(self):
            return 1_000

        def num_cached_edges(self):
            return 50_000

        def csr_view(self):
            self.called = True
            raise AssertionError("cap check happened too late")

    graph = ViewMustNotRun()
    with pytest.raises(MemoryError, match="vertices"):
        CMGDB.write_map_graph_csr_checkpoint(
            graph,
            tmp_path / "too-large.csr",
            configuration=_configuration(),
            caps=_caps(),
        )
    assert not graph.called
    assert not (tmp_path / "too-large.csr").exists()


def test_resume_caps_are_checked_before_mapping_arrays(tmp_path, monkeypatch):
    graph = _atlas_map_graph()
    path = tmp_path / "capped-load.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        path,
        configuration=_configuration(),
        caps=_caps(),
    )
    module = importlib.import_module("CMGDB.MapGraphCSR")

    def mapping_must_not_run(*_args, **_kwargs):
        raise AssertionError("resume cap check happened after np.load")

    monkeypatch.setattr(module.np, "load", mapping_must_not_run)
    with pytest.raises(MemoryError, match="vertices"):
        CMGDB.load_map_graph_csr_checkpoint(
            path,
            expected_configuration=_configuration(),
            caps=CMGDB.MapGraphCSRCheckpointCaps(
                max_vertices=2,
                max_edges=100,
                max_payload_bytes=10_000,
            ),
        )


def test_strict_resume_rejects_configuration_corruption_and_unknown_files(tmp_path):
    graph = _atlas_map_graph()

    mismatch = tmp_path / "mismatch.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        mismatch,
        configuration=_configuration(),
        caps=_caps(),
    )
    with pytest.raises(ValueError, match="configuration differs"):
        CMGDB.load_map_graph_csr_checkpoint(
            mismatch,
            expected_configuration={**_configuration(), "depth": 12},
            caps=_caps(),
        )

    corrupt = tmp_path / "corrupt.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        corrupt,
        configuration=_configuration(),
        caps=_caps(),
    )
    targets = np.load(corrupt / "targets.npy", mmap_mode="r+")
    targets[-1] = 0
    targets.flush()
    del targets
    with pytest.raises(ValueError, match="checksum|sorted"):
        CMGDB.load_map_graph_csr_checkpoint(
            corrupt,
            expected_configuration=_configuration(),
            caps=_caps(),
        )

    extra = tmp_path / "extra.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        extra,
        configuration=_configuration(),
        caps=_caps(),
    )
    (extra / "unexpected.txt").write_text("not part of the schema")
    with pytest.raises(ValueError, match="missing or unknown"):
        CMGDB.load_map_graph_csr_checkpoint(
            extra,
            expected_configuration=_configuration(),
            caps=_caps(),
        )


def test_strict_resume_rejects_truncated_mmap_artifact(tmp_path):
    graph = _atlas_map_graph()
    path = tmp_path / "truncated.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        path,
        configuration=_configuration(),
        caps=_caps(),
    )
    targets_path = path / "targets.npy"
    with targets_path.open("r+b") as stream:
        stream.truncate(targets_path.stat().st_size - 1)

    with pytest.raises(ValueError, match="file size"):
        CMGDB.load_map_graph_csr_checkpoint(
            path,
            expected_configuration=_configuration(),
            caps=_caps(),
        )


def test_checkpoint_refuses_overwrite_and_lazy_mapgraph(tmp_path, monkeypatch):
    graph = _atlas_map_graph()
    path = tmp_path / "existing.csr"
    CMGDB.write_map_graph_csr_checkpoint(
        graph,
        path,
        configuration=_configuration(),
        caps=_caps(),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        CMGDB.write_map_graph_csr_checkpoint(
            graph,
            path,
            configuration=_configuration(),
            caps=_caps(),
        )

    monkeypatch.setenv("CMGDB_MAPGRAPH_CACHE", "0")
    lazy_graph = _atlas_map_graph()
    assert not lazy_graph.has_cache()
    with pytest.raises(RuntimeError, match="CACHE"):
        CMGDB.write_map_graph_csr_checkpoint(
            lazy_graph,
            tmp_path / "lazy.csr",
            configuration=_configuration(),
            caps=_caps(),
        )


@pytest.mark.parametrize(
    ("variable", "limit", "message"),
    [
        ("CMGDB_MAPGRAPH_HARD_MAX_VERTICES", "2", "vertex count"),
        ("CMGDB_MAPGRAPH_HARD_MAX_EDGES", "2", "edge count"),
        ("CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES", "55", "edge/cache-byte"),
    ],
)
def test_native_hard_caps_stop_csr_growth(variable, limit, message, monkeypatch):
    monkeypatch.setenv(variable, limit)
    with pytest.raises(RuntimeError, match=message):
        _atlas_map_graph()


def test_native_hard_caps_allow_the_exact_cached_payload(monkeypatch):
    # Three vertices need four int64 offsets and this fixture has three
    # uint64 native edges: 4*8 + 3*8 = 56 bytes.
    monkeypatch.setenv("CMGDB_MAPGRAPH_HARD_MAX_VERTICES", "3")
    monkeypatch.setenv("CMGDB_MAPGRAPH_HARD_MAX_EDGES", "3")
    monkeypatch.setenv("CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES", "56")
    graph = _atlas_map_graph()
    assert graph.num_vertices() == 3
    assert graph.num_cached_edges() == 3


def test_native_hard_cap_rejects_malformed_value(monkeypatch):
    monkeypatch.setenv("CMGDB_MAPGRAPH_HARD_MAX_EDGES", "2.5")
    with pytest.raises(ValueError, match="nonnegative base-10 integer"):
        _atlas_map_graph()
