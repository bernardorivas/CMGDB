"""Regression tests for the live box maps BoxMap and BoxMapBatch, and for
the box images a Model takes from a map."""

import re

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


def sqrt_map(X):
    # Undefined (NaN) for coordinates above 1
    return np.sqrt(1.0 - X)


def sqrt_map_scalar(x):
    return list(sqrt_map(np.array([x]))[0])


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("mode, rect", [
    ("corners", [0.5, 0.5, 1.5, 1.5]),   # NaN at every corner but the first
    ("corners", [0.5, 1.2, 1.5, 1.5]),
    ("corners", [1.2, 1.2, 1.5, 1.5]),   # NaN at every corner
    ("center", [0.5, 1.2, 1.5, 1.5]),
    ("center", [1.2, 1.2, 1.5, 1.5]),
])
def test_nan_image_raises_in_both_box_maps(mode, rect):
    # C45: BoxMap's Python min/max skipped a NaN unless it came first, while
    # BoxMapBatch's ndarray.min/max propagated it, so the two disagreed on
    # these boxes; and a NaN bound cannot be covered
    with pytest.raises(ValueError, match="NaN"):
        CMGDB.BoxMap(sqrt_map_scalar, rect, mode=mode)
    with pytest.raises(ValueError, match="NaN"):
        CMGDB.BoxMapBatch(sqrt_map, np.array([[0.0, 0.0, 0.5, 0.5], rect]), mode=mode)


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("use_batch", [False, True])
def test_nan_image_stops_the_morse_graph_computation(use_batch):
    # C45: the two paths sent different NaN rectangles to C++, where the
    # cover of a NaN bound is undefined (a slab at the lower boundary on
    # arm64), and the batch map turned 1 Morse set into 5
    model = CMGDB.Model(4, 8, [0.0, 0.0], [1.5, 1.5],
                        lambda rect: CMGDB.BoxMap(sqrt_map_scalar, rect))
    if use_batch:
        model.set_batch_map(lambda rects: CMGDB.BoxMapBatch(sqrt_map, rects))
    with pytest.raises(ValueError, match="NaN"):
        CMGDB.ComputeMorseGraph(model)


def half_map_undefined_right(rects):
    """Box images of x/2 on [0, 1]^2, NaN for the boxes in x >= 0.5, as a
    map of the user's own might return them."""
    rects = np.asarray(rects, dtype=float)
    images = 0.5 * rects
    images[rects[:, 0] >= 0.5] = np.nan
    return images


@pytest.mark.parametrize("use_batch", [False, True])
def test_nan_image_from_any_map_raises(use_batch):
    # C45: a NaN bound from a map other than BoxMap or BoxMapBatch still
    # reached TreeGrid's cover, whose int64 cast of NaN is undefined (on
    # arm64 a slab of boxes at the lower boundary), and the run went on
    def F(rect):
        if use_batch:
            # Only the batch map returns NaN, so the error comes from there
            return [0.5 * v for v in rect]
        return half_map_undefined_right([rect])[0].tolist()

    model = CMGDB.Model(4, 4, [0.0, 0.0], [1.0, 1.0], F)
    if use_batch:
        model.set_batch_map(half_map_undefined_right)
    which = "The batch map" if use_batch else "The map"
    with pytest.raises(ValueError, match=which + " returned an image with a NaN bound") as info:
        CMGDB.ComputeMorseGraph(model)
    rect = [float(v) for v in re.search(r"\[(.*)\]", str(info.value)).group(1).split(", ")]
    assert len(rect) == 4 and rect[0] >= 0.5


def test_infinite_images_still_map():
    # An infinite image (an orbit escaping) is not an error
    def f(X):
        return 1.0 / X

    rect = [0.0, 0.25, 0.5, 0.5]
    expected = [2.0, 2.0, np.inf, 4.0]
    with np.errstate(divide="ignore"):
        assert CMGDB.BoxMap(lambda x: list(f(np.array([x]))[0]), rect) == expected
        assert CMGDB.BoxMapBatch(f, np.array([rect])).tolist() == [expected]
