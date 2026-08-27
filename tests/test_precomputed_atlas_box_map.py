from __future__ import annotations

import pytest

import CMGDB


def test_strict_lookup_preserves_tagged_union_and_explicit_empty_image():
    table = CMGDB.PrecomputedAtlasBoxMap(
        [
            ((10, [0.0, 0.5]), [(10, [0.1, 0.2]), (20, [0.7, 0.8])]),
            ((20, [0.0, 0.5]), []),
        ]
    )

    first = table(10, [0.0, 0.5])
    assert first == [(10, [0.1, 0.2]), (20, [0.7, 0.8])]
    first[0][1][0] = 999.0
    assert table(10, [0.0, 0.5]) == [
        (10, [0.1, 0.2]),
        (20, [0.7, 0.8]),
    ]
    assert table(20, [0.0, 0.5]) == []


def test_missing_source_raises_instead_of_becoming_an_open_exit():
    table = CMGDB.PrecomputedAtlasBoxMap(
        [((0, [0.0, 0.5]), [(0, [0.0, 0.5])])]
    )

    with pytest.raises(KeyError, match="not precomputed"):
        table(0, [0.5, 1.0])
    assert table.stats() == CMGDB.AtlasLookupStats(entries=1, hits=0, misses=1)


def test_exact_keys_reject_duplicate_sources_without_decimal_rounding():
    source = (0, [0.0, 0.5])
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        CMGDB.PrecomputedAtlasBoxMap(
            [(source, [(0, [0.0, 0.5])]), (source, [(0, [0.5, 1.0])])]
        )

    left = 0.5
    right = float.fromhex("0x1.0000000000001p-1")
    assert CMGDB.exact_atlas_source_key(0, [0.0, left]) != (
        CMGDB.exact_atlas_source_key(0, [0.0, right])
    )


def test_bounded_batch_precompute_and_provenance_passthrough():
    chunks = []
    provenance = {}
    sources = [(3, [float(k), float(k + 1)]) for k in range(5)]

    def batch(chunk):
        chunks.append(len(chunk))
        values = []
        for chart, bounds in chunk:
            provenance[(chart, tuple(bounds))] = {"source_width": bounds[1] - bounds[0]}
            values.append([(chart, bounds)])
        return values

    table = CMGDB.precompute_atlas_box_map(
        lambda _chart, _bounds: pytest.fail("scalar callback should not run"),
        sources,
        batch_size=2,
        batch_callback=batch,
        provenance_callback=lambda chart, bounds: provenance[(chart, tuple(bounds))],
    )

    assert chunks == [2, 2, 1]
    assert table.precompute_summary.source_count == 5
    assert table.precompute_summary.batch_calls == 3
    assert table.precompute_summary.scalar_calls == 0
    assert "no continuous-image enclosure claim" in table.precompute_summary.semantics
    assert table.provenance(3, [2.0, 3.0]) == {"source_width": 1.0}
    assert table.batch(sources[:2]) == [
        [(3, [0.0, 1.0])],
        [(3, [1.0, 2.0])],
    ]


def test_batched_callback_must_return_one_union_per_source():
    with pytest.raises(ValueError, match="wrong number"):
        CMGDB.precompute_atlas_box_map(
            lambda _chart, _bounds: [],
            [(0, [0.0, 0.5]), (0, [0.5, 1.0])],
            batch_size=2,
            batch_callback=lambda _sources: [[]],
        )


def test_precomputed_callback_runs_through_native_atlas_model():
    sources = [
        (10, [0.0, 0.5]),
        (10, [0.5, 1.0]),
        (20, [0.0, 0.5]),
        (20, [0.5, 1.0]),
    ]

    def tagged_union(chart, bounds):
        other = 20 if chart == 10 else 10
        midpoint = 0.5 * (bounds[0] + bounds[1])
        return [(chart, [midpoint, midpoint]), (other, [midpoint, midpoint])]

    table = CMGDB.precompute_atlas_box_map(tagged_union, sources, batch_size=2)
    model = CMGDB.AtlasModel(1)
    model.add_chart(10, [0.0], [1.0])
    model.add_chart(20, [0.0], [1.0])
    model.set_map(table)

    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert map_graph.num_vertices() == 4
    assert morse_graph.num_vertices() == 2
    assert table.stats().misses == 0
    # ComputeMorseGraph traverses the fixed Atlas map more than once; every
    # traversal is now a strict lookup instead of a callback reevaluation.
    assert table.stats().hits >= len(sources)
