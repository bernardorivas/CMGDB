"""Regression tests for the Model.set_batch_map binding.

The test map uses only IEEE-exact elementwise operations (division), so the
scalar and batched evaluations agree bit for bit. The computations pass
cache_transition_graph=True so that they call the batch map whatever
CMGDB_MAPGRAPH_CACHE says.
"""

import numpy as np
import pytest
import CMGDB


def f_vec(X):
    return X / (2.0 - X)


def F_scalar(rect):
    return CMGDB.BoxMap(lambda x: list(f_vec(np.array(x))), rect, padding=False)


def F_batch(rects):
    return CMGDB.BoxMapBatch(f_vec, rects, padding=False)


def build_model():
    return CMGDB.Model(4, 6, [0.0, 0.0], [1.2, 1.2], F_scalar)


def compute(model):
    return CMGDB.ComputeMorseGraph(model, cache_transition_graph=True,
                                   batch_chunk_size=16)


def signature(morse_graph, map_graph):
    """Morse sets and Morse graph edges, invariant under vertex relabeling."""
    num_vertices = morse_graph.num_vertices()
    morse_sets = [sorted(morse_graph.morse_set(v)) for v in range(num_vertices)]
    order = sorted(range(num_vertices), key=lambda v: morse_sets[v])
    relabel = {v: i for i, v in enumerate(order)}
    edges = sorted((relabel[u], relabel[w])
                   for u, w in morse_graph.edges_unreduced())
    return [morse_sets[v] for v in order], edges, map_graph.num_vertices()


def lower_upper(rects):
    images = F_batch(rects)
    dim = images.shape[1] // 2
    return images[:, :dim], images[:, dim:]


def test_batch_map_may_keep_its_input():
    # The binding used to hand the batch map a view of a C++ buffer that is
    # refilled for the next chunk and freed when the computation returns, so
    # an array kept by the batch map later held other rectangles or freed
    # memory (C25, C38).
    kept, snapshots = [], []

    def keeping_batch(rects):
        kept.append(rects)
        snapshots.append(np.array(rects))
        return F_batch(rects)

    model = build_model()
    model.set_batch_map(keeping_batch)
    compute(model)
    assert len(kept) > 1
    for rects, snapshot in zip(kept, snapshots):
        assert not rects.flags.writeable
        assert np.array_equal(rects, snapshot)


def test_batch_map_input_outlives_a_raising_call():
    # The frame of a batch map that raised stays reachable from the
    # traceback, as in a post-mortem debugger.
    snapshots = []

    def raising_batch(rects):
        snapshots.append(np.array(rects))
        if len(snapshots) == 3:
            raise ValueError("stop")
        return F_batch(rects)

    model = build_model()
    model.set_batch_map(raising_batch)
    with pytest.raises(ValueError, match="stop") as excinfo:
        compute(model)
    tb = excinfo.tb
    while tb.tb_frame.f_code.co_name != "raising_batch":
        tb = tb.tb_next
    assert np.array_equal(tb.tb_frame.f_locals["rects"], snapshots[-1])


ACCEPTED_RESULTS = {
    "array": F_batch,
    "Fortran-order array": lambda rects: np.asfortranarray(F_batch(rects)),
    "list of lists": lambda rects: F_batch(rects).tolist(),
    "list of row arrays": lambda rects: list(F_batch(rects)),
}

MISSHAPED_RESULTS = {
    "transposed": lambda rects: F_batch(rects).T.copy(),
    "transposed view": lambda rects: F_batch(rects).T,
    "vstack": lambda rects: np.vstack(lower_upper(rects)),
    "list of lower and upper rows":
        lambda rects: np.vstack(lower_upper(rects)).tolist(),
    "stack": lambda rects: np.stack(lower_upper(rects)),
    "stack on axis 1": lambda rects: np.stack(lower_upper(rects), axis=1),
    "row vector": lambda rects: F_batch(rects).reshape(1, -1),
    "flat array": lambda rects: F_batch(rects).ravel(),
    "flat list": lambda rects: F_batch(rects).ravel().tolist(),
    "flat column-major array": lambda rects: F_batch(rects).ravel(order="F"),
    "flat lower then upper bounds":
        lambda rects: np.concatenate([b.ravel() for b in lower_upper(rects)]),
}


@pytest.fixture(scope="module")
def scalar_signature():
    return signature(*compute(build_model()))


@pytest.mark.parametrize("name", sorted(ACCEPTED_RESULTS))
def test_accepted_batch_results_match_scalar(name, scalar_signature):
    model = build_model()
    model.set_batch_map(ACCEPTED_RESULTS[name])
    assert signature(*compute(model)) == scalar_signature


@pytest.mark.parametrize("name", sorted(MISSHAPED_RESULTS))
def test_misshaped_batch_result_raises(name):
    # Any result with count*2*dim values used to be read row by row, so a
    # transposed or vstacked result, or a flat one in another order, silently
    # mixed the bounds of different rectangles (C26, C39). Shapes other than
    # those of ACCEPTED_RESULTS are refused now, and so are flat results for
    # more than one rectangle, since their order cannot be checked.
    model = build_model()
    model.set_batch_map(MISSHAPED_RESULTS[name])
    with pytest.raises(RuntimeError, match="shape"):
        compute(model)


def test_flat_hstack_result_in_1d_raises():
    # In 1D, np.hstack of the (count,) arrays of lower and upper bounds gives
    # all lower bounds, then all upper bounds. It was read row by row and
    # paired the bounds of different rectangles (C26).
    def hstacked(rects):
        return np.hstack([f_vec(rects[:, 0]), f_vec(rects[:, 1])])

    model = CMGDB.Model(4, 6, [0.0], [1.2], F_scalar)
    model.set_batch_map(hstacked)
    with pytest.raises(RuntimeError, match=r"result of shape \(\d+,\)"):
        compute(model)


def test_flat_result_for_one_rectangle_matches_scalar(scalar_signature):
    # With one rectangle, every way of flattening its row gives the same
    # values, so a flat result is accepted when count == 1, as from a batch
    # map that squeezes its result.
    counts = []

    def squeezed(rects):
        counts.append(len(rects))
        return np.squeeze(F_batch(rects))

    model = build_model()
    model.set_batch_map(squeezed)
    assert signature(*compute(model)) == scalar_signature
    assert 1 in counts and max(counts) > 1


def test_misshaped_batch_result_error_names_both_shapes():
    counts = []

    def transposed(rects):
        counts.append(len(rects))
        return F_batch(rects).T

    model = build_model()
    model.set_batch_map(transposed)
    with pytest.raises(RuntimeError) as excinfo:
        compute(model)
    count = counts[-1]
    assert f"result of shape (4, {count})" in str(excinfo.value)
    assert f"(count, 2*dim) = ({count}, 4)" in str(excinfo.value)
