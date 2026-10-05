"""Regression tests for the live box maps BoxMap and BoxMapBatch."""

import numpy as np
import pytest
import CMGDB


def logistic(X):
    # A natural map of one variable: it returns shape (m,), not (m, 1)
    return 3.5 * X[:, 0] * (1.0 - X[:, 0])


def logistic_scalar(x):
    return [3.5 * x[0] * (1.0 - x[0])]


@pytest.mark.parametrize("mode", ["corners", "center"])
def test_batch_takes_one_dimensional_output_in_both_modes(mode):
    # C46: center mode used f's (m,) output as it came, so with the padding
    # of shape (N, 1) it broadcast to an (N, 2N) result
    rects = np.array([[0.0, 0.25], [0.25, 0.5], [0.5, 0.75]])
    batched = CMGDB.BoxMapBatch(logistic, rects, mode=mode)
    assert batched.shape == (3, 2)
    for i, rect in enumerate(rects):
        assert batched[i].tolist() == CMGDB.BoxMap(logistic_scalar, list(rect), mode=mode)


@pytest.mark.parametrize("mode", ["corners", "center"])
def test_batch_rejects_transposed_output(mode):
    # C46: corners mode reshaped any output of the right size, so the
    # (dim, m) array of np.array([g0(X), g1(X)]) came back scrambled
    def transposed(X):
        return np.array([X[:, 1], X[:, 0]])

    rects = np.array([[0.0, 0.0, 0.5, 0.25], [0.5, 0.25, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0]])
    with pytest.raises(ValueError, match="f must return an array of shape"):
        CMGDB.BoxMapBatch(transposed, rects, mode=mode)


def test_one_dimensional_center_batch_map_matches_scalar_map():
    # C46: through a model the (N, 2N) result failed the size check, after
    # its quadratic allocation
    def build():
        return CMGDB.Model(6, 8, [0.0], [1.0],
                           lambda rect: CMGDB.BoxMap(logistic_scalar, rect, mode="center"))

    def morse_sets(model):
        morse_graph = CMGDB.ComputeMorseGraph(model)[0]
        return sorted(sorted(morse_graph.morse_set(v))
                      for v in range(morse_graph.num_vertices()))

    model = build()
    model.set_batch_map(lambda rects: CMGDB.BoxMapBatch(logistic, rects, mode="center"))
    assert morse_sets(model) == morse_sets(build())
