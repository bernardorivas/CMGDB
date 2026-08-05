import itertools

import numpy as np
import pytest

import CMGDB
import CMGDB.PrecomputedBoxMap as precomputed


def vector_map(points):
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack([x + 2.0 * y, x - y])


def direct_corner_box_map(rect, *, padding=False):
    rect = np.asarray(rect, dtype=np.float64)
    dim = rect.size // 2
    lower = rect[:dim]
    upper = rect[dim:]
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    values = vector_map(corners)
    out_lower = values.min(axis=0)
    out_upper = values.max(axis=0)
    if padding:
        box_size = upper - lower
        out_lower = out_lower - box_size
        out_upper = out_upper + box_size
    return np.concatenate([out_lower, out_upper])


def test_public_make_precomputed_box_map_uniform_matches_direct_corner_map():
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=4,
        mode="uniform",
        padding=False,
    )

    rect = [-0.5, 0.0, 0.0, 0.5]

    np.testing.assert_allclose(box_map(rect), direct_corner_box_map(rect), atol=1e-12)


def test_adaptive_precomputed_matches_direct_corner_map_at_odd_depth_cell():
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=5,
        mode="adaptive",
        padding=False,
    )

    # At depth 5 in d=2, axis 0 is split into 8 cells and axis 1 into 4.
    rect = [-0.25, -0.5, 0.0, 0.0]

    np.testing.assert_allclose(box_map(rect), direct_corner_box_map(rect), atol=1e-12)


def test_uniform_and_adaptive_precomputed_are_equal_for_divisible_uniform_grid():
    lower = [-1.0, -1.0]
    upper = [1.0, 1.0]
    uniform = CMGDB.make_precomputed_box_map(
        vector_map, lower, upper, subdiv_max=4, mode="uniform", padding=True
    )
    adaptive = CMGDB.make_precomputed_box_map(
        vector_map, lower, upper, subdiv_max=4, mode="adaptive", padding=True
    )

    rect = [0.0, -0.5, 0.5, 0.0]

    np.testing.assert_array_equal(adaptive(rect), uniform(rect))


def test_precomputed_box_map_object_batches_rectangles_like_single_calls():
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=4,
        mode="adaptive",
        padding=False,
    )
    rects = [
        [-0.5, 0.0, 0.0, 0.5],
        [0.0, -0.5, 0.5, 0.0],
    ]
    expected = [box_map(rect) for rect in rects]
    assert hasattr(box_map, "batch")
    np.testing.assert_allclose(box_map.batch(rects), expected)


def test_precomputed_box_map_batch_can_be_installed_on_model():
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=4,
        mode="uniform",
        padding=False,
    )
    model = CMGDB.Model(4, 4, 4, 10000, [-1.0, -1.0], [1.0, 1.0], box_map)
    model.set_batch_map(box_map.batch)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    assert morse_graph.num_vertices() >= 1


def test_precompute_corner_grid_splits_evaluator_calls_into_bounded_chunks():
    sizes = []

    def recording_map(points):
        sizes.append(len(points))
        return vector_map(points)

    grid, out_dim = precomputed.precompute_corner_grid(
        recording_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        corners_per_axis=17,
        batch_points=64,
    )

    assert grid.shape == (17, 17, 2)
    assert out_dim == 2
    assert len(sizes) >= 2
    assert max(sizes) <= 64
    assert sum(sizes) == 289


def test_precompute_corner_grid_chunked_output_matches_one_shot():
    chunked, _ = precomputed.precompute_corner_grid(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        corners_per_axis=17,
        batch_points=64,
    )
    one_shot, _ = precomputed.precompute_corner_grid(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        corners_per_axis=17,
        batch_points=10_000,
    )

    np.testing.assert_array_equal(chunked, one_shot)


def test_precomputed_box_map_has_no_table_size_cap():
    """No ``max_table_points``: a lattice is built, never pre-refused.

    subdiv_max=10 in 2-D is a 33x33 = 1089-corner table, which the old default
    cap of 10_000_000 permitted but a lower configured cap refused. Sizing the
    table is the caller's call; one that does not fit fails on allocation.
    """
    box_map = CMGDB.make_precomputed_box_map(
        vector_map,
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=10,
        mode="adaptive",
    )

    out = box_map([-1.0, -1.0, -0.5, -0.5])
    assert len(out) == 4
    assert out[0] <= out[2] and out[1] <= out[3]


def test_uniform_mode_rejects_non_divisible_subdivision_depth():
    with pytest.raises(ValueError, match="divisible"):
        CMGDB.make_precomputed_box_map(
            vector_map,
            lower_bounds=[-1.0, -1.0],
            upper_bounds=[1.0, 1.0],
            subdiv_max=3,
            mode="uniform",
        )


def test_torch_device_auto_prefers_mps_then_cuda_then_cpu(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert precomputed.select_torch_device("auto").type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert precomputed.select_torch_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert precomputed.select_torch_device("auto").type == "cpu"


def test_torch_module_is_supported_without_making_torch_required():
    torch = pytest.importorskip("torch")

    class LinearMap(torch.nn.Module):
        def forward(self, x):
            return torch.column_stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]])

    box_map = CMGDB.make_precomputed_box_map(
        LinearMap(),
        lower_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        subdiv_max=4,
        mode="uniform",
        padding=False,
        batch_points=5,
        device="cpu",
    )

    rect = [-0.5, 0.0, 0.0, 0.5]
    expected = np.array([-0.5, -1.0, 0.5, 0.0])
    np.testing.assert_allclose(box_map(rect), expected, atol=1e-7, rtol=1e-7)
