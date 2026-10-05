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
