from __future__ import annotations

import math

import CMGDB
import pytest


def two_chart_model(depth: int = 2) -> CMGDB.AtlasModel:
    model = CMGDB.AtlasModel(depth)
    model.add_chart(10, [0.0], [1.0])
    model.add_chart(20, [0.0], [1.0])

    def tagged_union(chart_id, rect):
        midpoint = 0.5 * (rect[0] + rect[1])
        other_chart = 20 if chart_id == 10 else 10
        # Two genuinely separate image pieces.  A Euclidean hull cannot carry
        # the fact that one piece lives in another chart.
        return [
            (chart_id, [midpoint, midpoint]),
            (other_chart, [midpoint, midpoint]),
        ]

    model.set_map(tagged_union)
    return model


def test_fixed_depth_tagged_union_flows_through_native_scc_pipeline():
    model = two_chart_model(depth=2)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

    assert model.chart_ids() == [10, 20]
    assert map_graph.num_vertices() == 8
    assert morse_graph.num_vertices() == 4

    for node in morse_graph.vertices():
        boxes = morse_graph.morse_set_chart_boxes(node)
        assert {chart_id for chart_id, _bounds in boxes} == {10, 20}
        assert len(boxes) == 2
        # AtlasGeo now also supplies ordinary bounds for existing plotting.
        assert sorted(bounds for _chart_id, bounds in boxes) == sorted(
            morse_graph.morse_set_boxes(node)
        )


def test_adaptive_atlas_decomposition_preserves_chart_tags():
    model = CMGDB.AtlasModel(1, 3, 0, 10000)
    model.add_chart(0, [0.0], [1.0])
    model.add_chart(1, [0.0], [1.0])
    evaluated_widths = []

    def paired_midpoints(chart_id, rect):
        evaluated_widths.append(rect[1] - rect[0])
        midpoint = 0.5 * (rect[0] + rect[1])
        return [
            (chart_id, [midpoint, midpoint]),
            (1 - chart_id, [midpoint, midpoint]),
        ]

    model.set_map(paired_midpoints)
    morse_graph = CMGDB.ComputeMorseGraphOnly(model)

    # Vertices are reported at phase_subdiv_min, while the persistence search
    # evaluates the descendants down to phase_subdiv_max.
    assert morse_graph.num_vertices() == 2
    assert min(evaluated_widths) == pytest.approx(1.0 / 8.0)
    for node in morse_graph.vertices():
        assert {
            chart_id
            for chart_id, _bounds in morse_graph.morse_set_chart_boxes(node)
        } == {0, 1}


def test_atlas_cells_expose_chart_id_and_bounds():
    model = two_chart_model(depth=1)
    atlas = model.phaseSpace()
    atlas.subdivide()

    cells = [atlas.cell(index) for index in range(atlas.size())]
    assert [cell.chart_id for cell in cells] == [10, 10, 20, 20]
    assert cells[0].bounds == [0.0, 0.5]
    assert cells[-1].bounds == [0.5, 1.0]


def test_tagged_piece_dict_and_object_forms_are_accepted():
    model = CMGDB.AtlasModel(0)
    model.add_chart(3, [0.0], [1.0])

    def image(_chart_id, _rect):
        return [
            {"chart_id": 3, "bounds": [0.25, 0.25]},
            CMGDB.TaggedRectangle(3, [0.75, 0.75]),
        ]

    model.set_map(image)
    _morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert map_graph.adjacencies(0) == [0]


@pytest.mark.parametrize(
    "bad_piece, message",
    [
        ((999, [0.5, 0.5]), "unknown chart 999"),
        ((0, [0.5]), "expected 2"),
        ((0, [0.75, 0.25]), "lower bound greater"),
        ((0, [math.nan, 0.5]), "non-finite"),
    ],
)
def test_invalid_target_piece_fails_before_cover(bad_piece, message):
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    model.set_map(lambda _chart_id, _rect: [bad_piece])

    with pytest.raises(ValueError, match=message):
        CMGDB.ComputeMorseGraphOnly(model)


def test_empty_union_is_a_valid_empty_image():
    model = CMGDB.AtlasModel(0)
    model.add_chart(0, [0.0], [1.0])
    model.set_map(lambda _chart_id, _rect: [])

    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() == 0
    assert map_graph.adjacencies(0) == []


def test_atlas_model_makes_no_direct_chomp_claim():
    model = two_chart_model(depth=1)

    with pytest.raises(RuntimeError, match="not available directly on AtlasModel"):
        CMGDB.ComputeConleyMorseGraph(model)
    with pytest.raises(RuntimeError, match="not available directly on AtlasModel"):
        CMGDB.ComputeConleyMorseGraphOnly(model)


def test_ordinary_model_api_remains_available():
    model = CMGDB.Model(1, [0.0], [1.0], lambda rect: rect)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() >= 1
    assert map_graph.num_vertices() == 2
