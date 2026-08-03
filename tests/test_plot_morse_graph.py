import CMGDB


class CountingMorseGraph:
    def __init__(self):
        self.edge_calls = 0

    @staticmethod
    def vertices():
        return (0, 1, 2)

    def edges(self):
        self.edge_calls += 1
        return ((0, 1), (0, 2), (1, 2))

    @staticmethod
    def adjacencies(_vertex):
        raise AssertionError("PlotMorseGraph must not issue per-node adjacency queries")

    @staticmethod
    def annotations(vertex):
        return ("index",) if vertex == 1 else ()


def test_plot_morse_graph_caches_reduced_edges_once():
    morse_graph = CountingMorseGraph()

    source = CMGDB.PlotMorseGraph(morse_graph, clist=["#112233"])

    assert morse_graph.edge_calls == 1
    assert "0 -> 1;" in source.source
    assert "0 -> 2;" in source.source
    assert "1 -> 2;" in source.source
    assert '1 [label="1 : (index)"' in source.source
    assert "{rank=same; 2 };" in source.source
    assert "{rank=same; 0 };" in source.source
