import CMGDB


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
