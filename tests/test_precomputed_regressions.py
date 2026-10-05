"""Regression tests for the precomputed box maps: the PrecomputedBoxMap class
and the grid-layout factories of CMGDB.precomputed_grid."""

import copy
import itertools

import numpy as np
import pytest
import CMGDB
from CMGDB import precomputed_grid


# Domains on which lower + cells*side, the last lattice node, rounds above
# (1.2000000000000002) or below (0.2999999999999998) the upper bound
ROUNDING_DOMAINS = [([-1.0, -1.0], [1.2, 1.2]), ([-3.0, -3.0], [0.3, 0.3])]


def domain_checked_map(lower, upper, sampled):
    """A map defined on the closed domain only, like a bounded interpolant:
    evaluating it outside raises. Records the largest point of each call."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def f(X):
        X = np.asarray(X, dtype=float)
        if np.any(X < lower) or np.any(X > upper):
            raise ValueError("map evaluated outside the domain")
        sampled.append(X.max(axis=0))
        return 0.5 * X
    return f


@pytest.mark.parametrize("lower, upper", ROUNDING_DOMAINS)
@pytest.mark.parametrize("mode", ["corners", "center"])
def test_class_lattice_ends_at_upper_bound(lower, upper, mode):
    # C47: TreeGrid puts the upper face of the boundary boxes at upper_bounds
    # exactly, and live BoxMap samples f there; the table must too
    sampled = []
    CMGDB.PrecomputedBoxMap(domain_checked_map(lower, upper, sampled),
                            lower, upper, 10, mode=mode)
    assert np.array_equal(np.max(sampled, axis=0), upper)


@pytest.mark.parametrize("lower, upper", ROUNDING_DOMAINS)
@pytest.mark.parametrize("layout", ["adaptive", "uniform"])
@pytest.mark.parametrize("eval_mode", ["corners", "center", "random"])
def test_factory_lattice_ends_at_upper_bound(lower, upper, layout, eval_mode):
    # C47, in the fork's precompute_corner_grid
    sampled = []
    CMGDB.make_precomputed_box_map(domain_checked_map(lower, upper, sampled),
                                   lower, upper, subdiv_max=10, mode=layout,
                                   eval_mode=eval_mode)
    assert np.array_equal(np.max(sampled, axis=0), upper)


def test_corner_grid_ends_at_upper_bound_on_refined_axes_only():
    # C47; an axis with a single node keeps it at the lower bound
    grid, _ = precomputed_grid.precompute_corner_grid(
        lambda X: X, lower_bounds=[-1.0, -1.0], upper_bounds=[1.2, 1.2],
        corners_per_axis=[1, 3])
    assert np.array_equal(grid[0, :, 0], [-1.0, -1.0, -1.0])
    assert grid[0, 0, 1] == -1.0 and grid[0, -1, 1] == 1.2


def test_upper_face_box_matches_live_box_map():
    # C47: for a map with sqrt(ub - x), the overshooting node gave NaN images
    # on every box of the upper face
    lower, upper = [-1.0, -1.0], [1.2, 1.2]

    def f_vec(X):
        return np.column_stack([1.2 - 0.8 * np.sqrt(1.2 - X[:, 0]), 0.5 * X[:, 1]])

    def f_scalar(x):
        return list(f_vec(np.array([x]))[0])

    F = CMGDB.PrecomputedBoxMap(f_vec, lower, upper, 10)
    assert not np.isnan(F._table).any()
    side = F._finest_box_side
    rect = [upper[0] - side[0], lower[1], upper[0], lower[1] + side[1]]
    assert np.allclose(F(rect), CMGDB.BoxMap(f_scalar, rect), rtol=0, atol=1e-12)


# Rectangles that reach outside [0, 1]^2 by a whole cell (side 1/16 at
# subdiv_max=8) or more, or have a NaN bound
SIDE = 1.0 / 16.0
OUTSIDE_RECTS = [
    [-SIDE, 0.0, SIDE, SIDE],             # crosses the lower face
    [0.5, 0.5, 1.0 + 4 * SIDE, 1.0],      # crosses the upper face
    [-1.0, -1.0, 2.0, 2.0],               # contains the whole domain
    [1.0, 1.0, 1.0 + SIDE, 1.0 + SIDE],   # entirely outside
    [np.nan, 0.0, SIDE, SIDE],
]


def double(X):
    return 2.0 * np.asarray(X, dtype=float)


@pytest.mark.parametrize("rect", OUTSIDE_RECTS)
def test_class_rejects_box_outside_domain(rect):
    # C48: the lattice indices were clipped to the domain, so the image of a
    # smaller box came back without an error
    F = CMGDB.PrecomputedBoxMap(double, [0.0, 0.0], [1.0, 1.0], 8)
    with pytest.raises(ValueError, match="outside"):
        F(rect)
    with pytest.raises(ValueError, match="outside"):
        F.batch([[0.0, 0.0, SIDE, SIDE], rect])


@pytest.mark.parametrize("layout", ["adaptive", "uniform"])
@pytest.mark.parametrize("rect", OUTSIDE_RECTS)
def test_factories_reject_box_outside_domain(layout, rect):
    # C48, in the fork's factories
    box_map = CMGDB.make_precomputed_box_map(double, [0.0, 0.0], [1.0, 1.0],
                                             subdiv_max=8, mode=layout)
    with pytest.raises(ValueError, match="outside"):
        box_map(rect)
    with pytest.raises(ValueError, match="outside"):
        box_map.batch([[0.0, 0.0, SIDE, SIDE], rect])


def test_boxes_on_the_domain_faces_still_map():
    F = CMGDB.PrecomputedBoxMap(double, [0.0, 0.0], [1.0, 1.0], 8)
    rects = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, SIDE, SIDE],
             [1.0 - SIDE, 1.0 - SIDE, 1.0, 1.0]]
    expected = [double(r).tolist() for r in rects]
    assert [F(r) for r in rects] == expected
    assert F.batch(rects).tolist() == expected


def sqrt_map(X):
    # Undefined (NaN) for coordinates above 1
    return np.sqrt(1.0 - np.asarray(X, dtype=float))


def sqrt_map_scalar(x):
    return list(sqrt_map(np.array([x]))[0])


# Boxes of [0, 1.5]^2 at subdiv_max=8, side 0.09375: in corners mode the
# first is NaN at every corner but the lower one, the second at all of them
NAN_DOMAIN = ([0.0, 0.0], [1.5, 1.5])
NAN_RECTS = [[0.75, 0.75, 1.5, 1.5], [1.125, 1.125, 1.5, 1.5]]
FINITE_RECT = [0.0, 0.0, 0.75, 0.75]


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("mode", ["corners", "center"])
@pytest.mark.parametrize("rect", NAN_RECTS)
def test_class_raises_on_nan_image(mode, rect):
    # C45: BoxMap and BoxMapBatch raise when f is NaN at a sample point,
    # while the table returned NaN bounds, which C++ cannot cover
    F = CMGDB.PrecomputedBoxMap(sqrt_map, *NAN_DOMAIN, 8, mode=mode)
    with pytest.raises(ValueError, match="NaN"):
        F(rect)
    with pytest.raises(ValueError, match="NaN") as info:
        F.batch([FINITE_RECT, rect])
    assert str(rect) in str(info.value)
    # Boxes whose sample points are finite still map
    expected = CMGDB.BoxMap(sqrt_map_scalar, FINITE_RECT, mode=mode)
    assert F(FINITE_RECT) == expected
    assert F.batch([FINITE_RECT]).tolist() == [expected]


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("layout", ["adaptive", "uniform"])
@pytest.mark.parametrize("eval_mode", ["corners", "center", "random"])
@pytest.mark.parametrize("rect", NAN_RECTS)
def test_factories_raise_on_nan_image(layout, eval_mode, rect):
    # C45, in the fork's factories
    box_map = CMGDB.make_precomputed_box_map(sqrt_map, *NAN_DOMAIN, subdiv_max=8,
                                             mode=layout, eval_mode=eval_mode)
    with pytest.raises(ValueError, match="NaN"):
        box_map(rect)
    with pytest.raises(ValueError, match="NaN"):
        box_map.batch([FINITE_RECT, rect])
    assert len(box_map(FINITE_RECT)) == 4


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("factory", ["class", "function"])
@pytest.mark.parametrize("use_batch", [False, True])
def test_nan_image_stops_the_morse_graph_computation(factory, use_batch):
    # C45: the NaN bounds reached TreeGrid's cover, whose int64 cast of NaN
    # is undefined, and the run gave 5 Morse sets with the class and one of
    # 16 cells with the factory; BoxMap raises on this model
    if factory == "class":
        F = CMGDB.PrecomputedBoxMap(sqrt_map, *NAN_DOMAIN, 8)
    else:
        F = CMGDB.make_precomputed_box_map(sqrt_map, *NAN_DOMAIN, subdiv_max=8)
    model = CMGDB.Model(4, 8, *NAN_DOMAIN, F)
    if use_batch:
        model.set_batch_map(F.batch)
    with pytest.raises(ValueError, match="NaN"):
        CMGDB.ComputeMorseGraph(model)


def ratio_map(X):
    return X / (2.0 - X)


def ratio_map_scalar(x):
    return list(ratio_map(np.array([x]))[0])


def test_off_lattice_box_error_blames_neither_depth_nor_bounds():
    # C42: a rectangle off the lattice but wider than a finest cell is not a
    # box of the subdivision grid at any depth, so no subdiv_max serves it;
    # the error asked whether subdiv_max was deep enough. This one is from
    # the Conley phase of the adaptive run below, whose Model has the
    # table's bounds, so the error must not blame other bounds either
    F = CMGDB.PrecomputedBoxMap(ratio_map, [0.0, 0.0], [1.2, 1.2], 10)
    rect = [0.0, 0.875, 0.15, 1.0]
    for call in (lambda: F(rect), lambda: F.batch([[0.0, 0.0, 0.6, 0.6], rect])):
        with pytest.raises(ValueError, match="not a box of the subdivision grid") as info:
            call()
        message = str(info.value)
        assert "subdiv_max=" not in message and "same bounds" not in message
        assert str(rect) in message


def test_box_finer_than_lattice_error_names_subdiv_max():
    # C42: boxes of a deeper subdivision keep the subdiv_max hint, whether or
    # not their bounds fall on the lattice
    F = CMGDB.PrecomputedBoxMap(ratio_map, [0.0, 0.0], [1.2, 1.2], 10)
    side = F._finest_box_side
    for rect in ([0.0, 0.0, side[0] / 2, side[1]],
                 [side[0] / 2, 0.0, side[0], side[1]]):
        with pytest.raises(ValueError, match="finer than the subdiv_max lattice.*subdiv_max=10"):
            F(rect)


def conley_signature(morse_graph, map_graph):
    num_vertices = morse_graph.num_vertices()
    morse_sets = [sorted(morse_graph.morse_set(v)) for v in range(num_vertices)]
    order = sorted(range(num_vertices), key=lambda v: morse_sets[v][0])
    return {
        "sizes": [len(morse_sets[v]) for v in order],
        "min_box": [morse_sets[v][0] for v in order],
        "phase_size": map_graph.num_vertices(),
        "conley": [tuple(morse_graph.annotations(v)) for v in order],
    }


@pytest.mark.parametrize("use_batch", [False, True])
def test_conley_morse_graph_on_adaptive_grid_matches_live(use_batch):
    # C42: the adaptive (6, 10, 4) run of tests/test_conley_batch.py
    lower, upper = [0.0, 0.0], [1.2, 1.2]

    def build(F):
        return CMGDB.Model(6, 10, 4, 10000, lower, upper, F)

    live = conley_signature(*CMGDB.ComputeConleyMorseGraph(
        build(lambda rect: CMGDB.BoxMap(ratio_map_scalar, rect))))
    F = CMGDB.PrecomputedBoxMap(ratio_map, lower, upper, 10)
    model = build(F)
    if use_batch:
        model.set_batch_map(F.batch)
    assert conley_signature(*CMGDB.ComputeConleyMorseGraph(model)) == live


def train_mode_net(torch):
    """A network in the middle of training, with one submodule the user froze
    in eval mode."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.BatchNorm1d(8),
                              torch.nn.Dropout(0.5), torch.nn.Tanh(),
                              torch.nn.Linear(8, 2))
    net.train()
    net[1].eval()
    return net


def module_state(net):
    devices = [t.device for t in itertools.chain(net.parameters(), net.buffers())]
    return devices, [m.training for m in net.modules()]


def lattice_points(F):
    """The nodes of F's table, in table order."""
    axes = [np.linspace(lo, up, n) for lo, up, n in
            zip(F.lower_bounds, F.upper_bounds, F._nodes_per_axis)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, F.dim)


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_class_leaves_torch_module_as_it_was(device):
    # C40: the module was moved to device and put in eval mode in place, so
    # the caller's next CPU call or training step failed or ran without
    # dropout
    torch = pytest.importorskip("torch")
    net = train_mode_net(torch)
    before = module_state(net)
    F = CMGDB.PrecomputedBoxMap(net, [0.0, 0.0], [1.0, 1.0], 6, device=device)
    assert module_state(net) == before
    if device == "cpu":
        # The table still comes from the network in eval mode
        reference = copy.deepcopy(net).eval()
        with torch.no_grad():
            expected = reference(torch.as_tensor(lattice_points(F), dtype=torch.float32))
        assert np.array_equal(F._table.reshape(-1, 2), expected.numpy().astype(float))


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_factories_leave_torch_module_as_it_was(device):
    # C40, in the fork's as_batched_evaluator
    torch = pytest.importorskip("torch")
    net = train_mode_net(torch)
    before = module_state(net)
    CMGDB.make_precomputed_box_map(net, [0.0, 0.0], [1.0, 1.0], subdiv_max=6,
                                   device=device)
    assert module_state(net) == before
    evaluator = precomputed_grid.as_batched_evaluator(net, device=device)
    evaluator(np.zeros((3, 2)))
    assert module_state(net) == before


def test_factory_lends_torch_module_once_per_table():
    # C40: the evaluator lends the module for each call, which on an
    # accelerator moves it to the device and back every time, and the
    # table build called it once per chunk
    torch = pytest.importorskip("torch")
    moves = []

    class CountingNet(torch.nn.Sequential):
        def to(self, *args, **kwargs):
            moves.append(args)
            return super().to(*args, **kwargs)

    torch.manual_seed(0)
    net = CountingNet(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    # 81 nodes, in 11 chunks of at most 8
    CMGDB.make_precomputed_box_map(net, [0.0, 0.0], [1.0, 1.0], subdiv_max=6,
                                   batch_points=8, device="cpu")
    assert len(moves) == 2
    moves.clear()
    precomputed_grid.as_batched_evaluator(net, device="cpu")(np.zeros((3, 2)))
    assert len(moves) == 2


def typed_net(torch, dtype):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(),
                               torch.nn.Linear(8, 2)).to(dtype)


def module_values(torch, net, points, dtype):
    """net evaluated on the CPU in its own dtype, as float64."""
    with torch.no_grad():
        return net(torch.as_tensor(points, dtype=dtype)).to(torch.float64).numpy()


@pytest.mark.parametrize("dtype_name", ["float64", "float16", "bfloat16"])
def test_class_evaluates_module_in_its_dtype(dtype_name):
    # C44: the points were always cast to float32, so a float64 (or half
    # precision) module failed in its first layer
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    net = typed_net(torch, dtype)
    F = CMGDB.PrecomputedBoxMap(net, [0.0, 0.0], [1.0, 1.0], 6, device="cpu")
    expected = module_values(torch, net, lattice_points(F), dtype)
    assert np.array_equal(F._table.reshape(-1, 2), expected)


@pytest.mark.parametrize("dtype_name", ["float64", "float16", "bfloat16"])
def test_factory_evaluates_module_in_its_dtype(dtype_name):
    # C44, in the fork's as_batched_evaluator
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    net = typed_net(torch, dtype)
    evaluator = precomputed_grid.as_batched_evaluator(net, device="cpu")
    points = np.random.default_rng(0).uniform(size=(16, 2))
    assert np.array_equal(evaluator(points), module_values(torch, net, points, dtype))


def test_auto_device_passes_over_mps_for_float64(monkeypatch):
    # C44: MPS has no float64, and 'auto' chose it anyway
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    net = typed_net(torch, torch.float64)
    F = CMGDB.PrecomputedBoxMap(net, [0.0, 0.0], [1.0, 1.0], 6, device="auto")
    expected = module_values(torch, net, lattice_points(F), torch.float64)
    assert np.array_equal(F._table.reshape(-1, 2), expected)
    assert precomputed_grid.select_torch_device("auto", dtype=torch.float64).type == "cpu"
    assert precomputed_grid.select_torch_device("auto", dtype=torch.float32).type == "mps"
    box_map = CMGDB.make_precomputed_box_map(net, [0.0, 0.0], [1.0, 1.0],
                                             subdiv_max=6, device="auto")
    assert len(box_map([0.0, 0.0, 0.125, 0.125])) == 4
