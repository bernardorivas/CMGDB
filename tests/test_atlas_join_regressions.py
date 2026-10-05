"""Regression tests for joining Atlas grids.

Compute_Morse_Graph joins a decomposition node's grid into itself, so the
output of joinImpl<Atlas> is also one of its inputs. The join cleared the
output before reading the inputs, so an adaptive AtlasModel run lost its
charts: it returned no cells, or raised "Atlas::join error: just pushed back
a nonconformant chart". A one-chart AtlasModel must give the Morse graph of
the Model with the same bounds, subdivision and map.
"""

import pytest
import CMGDB


def f(x):
    return x + 0.5 * x * (1.0 - x * x)


def image(lower, upper):
    a, b = f(lower), f(upper)
    return [min(a, b), max(a, b)]


def model_summary(morse_graph):
    return sorted(
        (tuple(sorted(tuple(box) for box in morse_graph.morse_set_boxes(v))),
         tuple(sorted(morse_graph.adjacencies(v))))
        for v in range(morse_graph.num_vertices()))


def atlas_summary(morse_graph):
    return sorted(
        (tuple(sorted(tuple(bounds) for _, bounds in morse_graph.morse_set_chart_boxes(v))),
         tuple(sorted(morse_graph.adjacencies(v))))
        for v in range(morse_graph.num_vertices()))


@pytest.mark.parametrize("subdiv", [(4, 8, 0), (6, 10, 0), (6, 10, 4),
                                    (8, 8, 0), (8, 12, 4), (10, 10, 10)])
def test_one_chart_atlas_matches_model(subdiv):
    subdiv_min, subdiv_max, subdiv_init = subdiv
    atlas = CMGDB.AtlasModel(subdiv_min, subdiv_max, subdiv_init)
    atlas.add_chart(0, [-1.5], [1.5])
    atlas.set_map(lambda chart, rect: [(0, image(rect[0], rect[1]))])
    atlas_mg, atlas_graph = CMGDB.ComputeMorseGraph(atlas)

    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, 10000,
                        [-1.5], [1.5], lambda rect: image(rect[0], rect[1]))
    model_mg, model_graph = CMGDB.ComputeMorseGraph(model)

    assert atlas_graph.num_vertices() == model_graph.num_vertices()
    assert atlas_mg.num_vertices() == model_mg.num_vertices() > 0
    assert atlas_summary(atlas_mg) == model_summary(model_mg)


def shifted_image(lower, upper, shift):
    a, b = image(lower - shift, upper - shift)
    return [a + shift, b + shift]


def labeled_morse_graph(sets, edges):
    # Morse sets as frozensets of boxes, edges as pairs of such sets, so that
    # graphs with different vertex numberings compare
    return ({frozenset(s) for s in sets},
            {(frozenset(sets[u]), frozenset(sets[v])) for u, v in edges})


@pytest.mark.parametrize("subdiv", [(4, 8, 0), (6, 10, 4), (8, 8, 0)])
def test_two_chart_atlas_matches_two_models(subdiv):
    # Two disjoint charts, each mapped into itself: the Morse graph of the
    # atlas is the disjoint union of the Morse graphs of two Models.
    subdiv_min, subdiv_max, subdiv_init = subdiv
    shifts = {0: 0.0, 1: 11.5}
    atlas = CMGDB.AtlasModel(subdiv_min, subdiv_max, subdiv_init)
    for chart, shift in shifts.items():
        atlas.add_chart(chart, [shift - 1.5], [shift + 1.5])
    atlas.set_map(lambda chart, rect:
                  [(chart, shifted_image(rect[0], rect[1], shifts[chart]))])
    atlas_mg, atlas_graph = CMGDB.ComputeMorseGraph(atlas)
    atlas_sets = [tuple(sorted((chart, tuple(bounds)) for chart, bounds
                               in atlas_mg.morse_set_chart_boxes(v)))
                  for v in range(atlas_mg.num_vertices())]
    atlas_edges = [(u, v) for u in range(atlas_mg.num_vertices())
                   for v in atlas_mg.adjacencies(u)]

    model_sets, model_edges, model_cells = [], [], 0
    for chart, shift in shifts.items():
        model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, 10000,
                            [shift - 1.5], [shift + 1.5],
                            lambda rect, s=shift: shifted_image(rect[0], rect[1], s))
        mg, graph = CMGDB.ComputeMorseGraph(model)
        offset = len(model_sets)
        model_sets += [tuple(sorted((chart, tuple(box)) for box in mg.morse_set_boxes(v)))
                       for v in range(mg.num_vertices())]
        model_edges += [(offset + u, offset + v) for u in range(mg.num_vertices())
                        for v in mg.adjacencies(u)]
        model_cells += graph.num_vertices()

    assert atlas_graph.num_vertices() == model_cells
    assert len(atlas_sets) == len(model_sets) > 0
    assert (labeled_morse_graph(atlas_sets, atlas_edges)
            == labeled_morse_graph(model_sets, model_edges))
