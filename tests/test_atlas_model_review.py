from __future__ import annotations

import CMGDB


def test_disjoint_pieces_in_one_chart_are_not_convexified():
    model = CMGDB.AtlasModel(2)
    model.add_chart(0, [0.0], [1.0])
    model.set_map(
        lambda _chart, _bounds: [
            (0, [0.1, 0.1]),
            (0, [0.9, 0.9]),
        ]
    )

    _morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

    # At depth two the four cells are [0,.25], [.25,.5], [.5,.75],
    # and [.75,1].  A hull of the two pieces would hit all four cells.
    assert map_graph.adjacencies(0) == [0, 3]


def test_atlas_does_not_infer_a_missing_quotient_face_representative():
    def adjacencies(return_both_faces: bool):
        model = CMGDB.AtlasModel(1)
        model.add_chart(0, [0.0], [1.0])  # base chart; seam at x=1
        model.add_chart(1, [0.0], [1.0])  # handle chart; seam at s=0

        def box_map(_source_chart, _source_bounds):
            pieces = [(0, [1.0, 1.0])]
            if return_both_faces:
                pieces.append((1, [0.0, 0.0]))
            return pieces

        model.set_map(box_map)
        _morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
        return map_graph.adjacencies(0)

    assert adjacencies(return_both_faces=False) == [1]
    assert adjacencies(return_both_faces=True) == [1, 2]


def test_mixed_chart_dimensions_survive_adaptive_join_and_extraction():
    model = CMGDB.AtlasModel(1, 2, 0, 10_000)
    model.add_chart(10, [0.0], [1.0])
    model.add_chart(20, [0.0, 0.0], [1.0, 1.0])

    def box_map(chart_id, bounds):
        dimension = len(bounds) // 2
        midpoint = [
            0.5 * (bounds[d] + bounds[dimension + d])
            for d in range(dimension)
        ]
        if chart_id == 10:
            return [
                (10, [midpoint[0], midpoint[0]]),
                (20, [midpoint[0], 0.5, midpoint[0], 0.5]),
            ]
        return [
            (10, [midpoint[0], midpoint[0]]),
            (20, midpoint + midpoint),
        ]

    model.set_map(box_map)
    morse_graph = CMGDB.ComputeMorseGraphOnly(model)

    assert morse_graph.num_vertices() == 2
    for node in morse_graph.vertices():
        boxes = morse_graph.morse_set_chart_boxes(node)
        assert {chart_id for chart_id, _bounds in boxes} == {10, 20}
        assert {len(bounds) for _chart_id, bounds in boxes} == {2, 4}
