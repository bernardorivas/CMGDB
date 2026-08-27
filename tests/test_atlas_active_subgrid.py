from __future__ import annotations

import CMGDB
import pytest


def _mixed_sparse_model() -> CMGDB.AtlasModel:
    model = CMGDB.AtlasModel(0)
    model.add_chart(10, [0.0, 0.0], [1.0, 1.0])
    model.add_chart(20, [-1.0], [1.0])
    model.add_chart(30, [0.0], [1.0])
    model.set_active_subgrid(
        [
            (10, 2, [0, 0]),
            {"chart_id": 10, "axis_depth": 2, "coordinates": [3, 3]},
            CMGDB.TaggedDyadicCell(20, 3, [3]),
            # Exact duplicates are harmless and canonicalized.
            (20, 3, [3]),
        ]
    )
    return model


def test_selected_dyadic_cells_preserve_mixed_charts_and_omit_full_grid():
    model = _mixed_sparse_model()
    atlas = model.phaseSpace()

    assert model.chart_ids() == [10, 20, 30]
    assert model.active_subgrid_configured()
    assert model.initial_cell_count() == 3
    assert atlas.num_charts() == 3
    assert atlas.size() == 3
    assert [atlas.cell(index).chart_id for index in range(3)] == [10, 10, 20]
    assert atlas.cell(0).bounds == pytest.approx([0.0, 0.0, 0.25, 0.25])
    assert atlas.cell(1).bounds == pytest.approx([0.75, 0.75, 1.0, 1.0])
    assert atlas.cell(2).bounds == pytest.approx([-0.25, 0.0])

    declared = model.active_dyadic_cells()
    assert [
        (cell.chart_id, cell.axis_depth, cell.coordinates) for cell in declared
    ] == [
        (10, 2, [0, 0]),
        (10, 2, [3, 3]),
        (20, 3, [3]),
    ]

    # Cover works on the sparse active geometry itself.  A point in the
    # ambient chart but outside the active family is an explicit exit.
    assert atlas.cover(10, [0.1, 0.1, 0.1, 0.1]) == [0]
    assert atlas.cover(10, [0.5, 0.5, 0.5, 0.5]) == []
    assert atlas.cover(30, [0.5, 0.5]) == []


def test_depth_thirty_cell_is_constructed_without_a_full_2_to_60_grid():
    model = CMGDB.AtlasModel(0)
    model.add_chart(7, [0.0, 0.0], [1.0, 1.0])
    model.set_active_subgrid([(7, 30, [123456789, 987654321])])

    assert model.initial_cell_count() == 1
    cell = model.phaseSpace().cell(0)
    scale = 2**30
    assert cell.bounds == pytest.approx(
        [
            123456789 / scale,
            987654321 / scale,
            123456790 / scale,
            987654322 / scale,
        ]
    )


@pytest.mark.parametrize(
    "cells, message",
    [
        ([(999, 1, [0])], "unknown chart 999"),
        ([(0, 1, [0, 0])], "needs 1 coordinate"),
        ([(0, 2, [4])], "outside"),
        ([(0, 64, [0])], "at most 63"),
        ([(0, 0, [0]), (0, 1, [0])], "antichain"),
    ],
)
def test_active_family_validation_is_transactional(cells, message):
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    before = model.phaseSpace().cell(0).bounds

    with pytest.raises(ValueError, match=message):
        model.set_active_subgrid(cells)

    assert not model.active_subgrid_configured()
    assert model.initial_cell_count() == 1
    assert model.phaseSpace().cell(0).bounds == before


def test_active_family_locks_chart_structure_and_map_locks_active_family():
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    model.set_active_subgrid([(0, 2, [1])])
    with pytest.raises(RuntimeError, match="cannot be changed"):
        model.add_chart(1, [0.0], [1.0])

    model.set_map(lambda _chart, _bounds: [])
    with pytest.raises(RuntimeError, match="after set_map"):
        model.set_active_subgrid([(0, 2, [2])])


def test_mapgraph_and_morse_semantics_distinguish_retained_targets_from_exits():
    model = _mixed_sparse_model()
    callback_kinds: dict[int, str] = {}

    def box_map(chart_id, bounds):
        if chart_id == 20:
            callback_kinds[chart_id] = "explicit-empty"
            return []
        if bounds[0] < 0.5:
            callback_kinds[0] = "active-target"
            return [(10, [0.1, 0.1, 0.1, 0.1])]
        callback_kinds[1] = "inactive-target-exit"
        # This is a nonempty geometric image, but it lies in the inactive gap.
        return [(10, [0.5, 0.5, 0.5, 0.5])]

    model.set_map(box_map)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

    assert map_graph.num_vertices() == 3
    assert map_graph.adjacencies(0) == [0]
    assert map_graph.adjacencies(1) == []  # inactive-family boundary exit
    assert map_graph.adjacencies(2) == []  # callback's explicit empty image
    assert set(callback_kinds.values()) == {
        "active-target",
        "inactive-target-exit",
        "explicit-empty",
    }
    # CMGDB's ordinary SCC/Morse pipeline runs on the active Atlas: only the
    # retained self-loop is recurrent.
    assert morse_graph.num_vertices() == 1
    assert morse_graph.morse_set_chart_boxes(0) == [
        (10, pytest.approx([0.0, 0.0, 0.25, 0.25]))
    ]


def test_empty_active_family_is_an_empty_atlas_and_morse_graph():
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    model.add_chart(1, [0.0], [1.0])
    model.set_active_subgrid([])
    model.set_map(lambda _chart, _bounds: [])

    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert model.phaseSpace().num_charts() == 2
    assert map_graph.num_vertices() == 0
    assert morse_graph.num_vertices() == 0
