"""Regression tests for the Morse set plots (review findings C01, C02 and C04-C22)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import CMGDB

H = 0.1


def corner_rows():
    """A 5x5 block (set 0), a 3x3 block beside it (set 1), one box in the corner (set 2)."""
    rows = [[i * H, j * H, (i + 1) * H, (j + 1) * H, 0] for i in range(5) for j in range(5)]
    rows += [[i * H, j * H, (i + 1) * H, (j + 1) * H, 1] for i in range(5, 8) for j in range(3)]
    rows.append([0.9, 0.9, 1.0, 1.0, 2])
    return rows


def drawn_vertices(ax):
    """Every vertex of the patches and collections drawn on ax, in data units."""
    paths = [patch.get_path().transformed(patch.get_patch_transform())
             for patch in ax.patches]
    for collection in ax.collections:
        paths.extend(collection.get_paths())
    return np.concatenate([path.vertices for path in paths])


@pytest.mark.parametrize("scale_factor", [[1, 1, 10], [10, 1], [1, 10], [3, 1, 1]])
def test_axis_limits_contain_the_inflated_sets(scale_factor):
    # C01: the limits looked the factor up by value, not by node, and clipped
    # every inflated set whose node number was not also one of the factors.
    fig, ax = CMGDB.PlotMorseSets(corner_rows(), scale_factor=scale_factor, show=False)
    vertices = drawn_vertices(ax)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert x0 <= vertices[:, 0].min() and vertices[:, 0].max() <= x1
    assert y0 <= vertices[:, 1].min() and vertices[:, 1].max() <= y1
    plt.close(fig)


def tiny_rows():
    """Three sets of one 0.01 box each, at 0, 1 and 0.5 on the diagonal."""
    return [[0.0, 0.0, 0.01, 0.01, 0], [1.0, 1.0, 1.01, 1.01, 1],
            [0.5, 0.5, 0.51, 0.51, 2]]


def patch_width(patch):
    vertices = patch.get_path().vertices
    return vertices[:, 0].max() - vertices[:, 0].min()


def test_dict_scale_factor_is_read_by_node():
    # C20, C01: a dict was padded with list(), which made its keys the
    # factors: {2: 10} became [2, 1, 1], inflating set 0 and not set 2.
    fig, ax = CMGDB.PlotMorseSets(tiny_rows(), scale_factor={2: 10}, show=False)
    assert [patch_width(p) for p in ax.patches] == pytest.approx([0.01, 0.01, 0.1])
    plt.close(fig)
    fig, ax = CMGDB.PlotMorseSets(tiny_rows(), morse_nodes=[2], scale_factor={2: 10},
                                  show=False)
    assert [patch_width(p) for p in ax.patches] == pytest.approx([0.1])
    plt.close(fig)
    # Key 0 used to become a factor of 0, and the set vanished.
    fig, ax = CMGDB.PlotBoxesScatter(tiny_rows(), morse_nodes=[0], scale_factor={0: 5},
                                     show=False)
    fig_l, ax_l = CMGDB.PlotBoxesScatter(tiny_rows(), morse_nodes=[0],
                                         scale_factor=[5, 1, 1], show=False)
    assert ax.collections[0].get_sizes() == pytest.approx(ax_l.collections[0].get_sizes())
    assert ax.collections[0].get_sizes()[0] > 0
    plt.close(fig)
    plt.close(fig_l)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.longdouble])
def test_numpy_scale_factors_merge_like_floats(dtype):
    # C21: Fraction() takes no numpy float that is not a float subclass
    # (before Python 3.14), so these factors raised TypeError on the default
    # merged path.
    rows = [[i * H, 0, (i + 1) * H, H, 0] for i in range(5)] + [[1, 1, 1 + H, 1 + H, 1]]
    fig, ax = CMGDB.PlotMorseSets(rows, scale_factor=list(np.array([2.0, 1.0], dtype=dtype)),
                                  show=False)
    fig_f, ax_f = CMGDB.PlotMorseSets(rows, scale_factor=[2.0, 1.0], show=False)
    assert len(ax.patches) == 2 and len(ax.collections) == 0
    for patch, reference in zip(ax.patches, ax_f.patches):
        assert np.allclose(patch.get_path().vertices, reference.get_path().vertices)
    plt.close(fig)
    plt.close(fig_f)


def test_boxes_scatter_takes_one_dimensional_boxes():
    # C04: 1.3.2's PlotBoxesScatter lifted 1-D boxes to two dimensions itself;
    # the lift had moved to PlotMorseSetsScatter, so a direct call failed its
    # projection assertion.
    rows = [[0.0, 0.1, 0], [0.1, 0.2, 0], [0.5, 0.6, 1]]
    fig, ax = CMGDB.PlotBoxesScatter(rows, show=False)
    # Each box [a, b] is drawn as the square [a, b] x [0, b - a].
    assert len(ax.collections) == 2
    assert np.allclose(ax.collections[0].get_offsets(), [[0.05, 0.05], [0.15, 0.05]])
    assert np.allclose(ax.collections[1].get_offsets(), [[0.55, 0.05]])
    plt.close(fig)


def line_rows():
    """Set 0: three touching boxes on [0.40, 0.43]; set 1: one box at 0.9."""
    return [[0.40, 0.41, 0], [0.41, 0.42, 0], [0.42, 0.43, 0], [0.9, 0.91, 1]]


def x_extents(collection):
    """The x extent of each path of a collection, rounded."""
    return sorted((round(float(path.vertices[:, 0].min()), 9),
                   round(float(path.vertices[:, 0].max()), 9))
                  for path in collection.get_paths())


def test_one_dimensional_dispatch_forwards_scale_and_edges():
    # C05: PlotMorseSets dropped scale_factor, edge_clr and linewidth when it
    # handed 1-D data to PlotMorseSets1D, so they were ignored without a word.
    kwargs = dict(scale_factor=[5, 1], edge_clr='k', linewidth=3.0)
    fig, ax = CMGDB.PlotMorseSets(line_rows(), show=False, **kwargs)
    fig_1, ax_1 = CMGDB.PlotMorseSets1D(line_rows(), show=False, **kwargs)
    for mine, reference in zip(ax.collections, ax_1.collections):
        assert x_extents(mine) == x_extents(reference)
        assert np.allclose(mine.get_edgecolor(), [[0, 0, 0, 1]])
        assert np.allclose(mine.get_linewidth(), 3.0)
    low, high = x_extents(ax.collections[0])[0]
    assert low < 0.40 and high > 0.43           # set 0 is drawn inflated
    plt.close(fig)
    plt.close(fig_1)


def set_labels(ax):
    """The set numbers printed over the 1-D pieces, with their x positions."""
    return sorted((t.get_text(), round(t.get_position()[0], 9)) for t in ax.texts
                  if t.get_text().isdigit())


def test_one_dimensional_scale_factor_scales_each_box():
    # C06: a piece was scaled as a whole, so three touching boxes inflated 5x
    # covered 15 boxes, [0.34, 0.49], where the per-box union every other plot
    # draws is [0.38, 0.45].
    fig, ax = CMGDB.PlotMorseSets1D(line_rows(), scale_factor=[5, 1], show=False)
    assert x_extents(ax.collections[0]) == [(0.38, 0.45)]
    assert set_labels(ax) == [('0', 0.415), ('1', 0.905)]
    plt.close(fig)
    # Shrunk boxes come apart; the piece still has one label.
    fig, ax = CMGDB.PlotMorseSets1D(line_rows(), scale_factor=[0.5, 1], show=False)
    assert x_extents(ax.collections[0]) == [(0.4025, 0.4075), (0.4125, 0.4175),
                                            (0.4225, 0.4275)]
    assert set_labels(ax) == [('0', 0.415), ('1', 0.905)]
    plt.close(fig)


def test_one_dimensional_labels_stay_inside_an_explicit_xlim(tmp_path):
    # C07: every piece was labeled, including pieces outside an explicit
    # xlim. Text is not clipped, so those labels were drawn beside the axes
    # and the tight bounding box of a saved figure stretched to take them in.
    rows = [[0.10, 0.20, 0], [0.70, 0.80, 1], [2.0, 2.1, 2]]
    fig, ax = CMGDB.PlotMorseSets1D(rows, xlim=[0, 0.5], show=False)
    assert set_labels(ax) == [('0', 0.15)]
    plt.close(fig)
    # A piece the window cuts is labeled at the middle of what it shows.
    fig, ax = CMGDB.PlotMorseSets1D(rows, xlim=[0.15, 0.75], show=False)
    assert set_labels(ax) == [('0', 0.175), ('1', 0.725)]
    plt.close(fig)
    widths = []
    for label_sets in (True, False):
        out = tmp_path / f'labels_{label_sets}.png'
        fig, ax = CMGDB.PlotMorseSets1D(rows, xlim=[0, 0.5], label_sets=label_sets,
                                        fig_fname=str(out), dpi=80, show=False)
        widths.append(plt.imread(out).shape[1])
        plt.close(fig)
    assert abs(widths[0] - widths[1]) <= 2


@pytest.mark.parametrize("label_sets", [True, False])
@pytest.mark.parametrize("height", [0.18, 1.2, 2.0])
def test_one_dimensional_limits_follow_the_box_height(height, label_sets):
    # C08: ylim was fixed at [-0.5, 0.78], or [-0.5, 0.5] without labels, so
    # a box taller than 1 was cut off at the bottom of the axes.
    fig, ax = CMGDB.PlotMorseSets1D(line_rows(), height=height, label_sets=label_sets,
                                    show=False)
    low, high = ax.get_ylim()
    assert low < -height / 2 and height / 2 < high
    if height == 0.18:                       # the default layout is unchanged
        assert (low, high) == pytest.approx((-0.5, 0.78 if label_sets else 0.5))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = ax.get_window_extent(renderer).y1
    for text in ax.texts:
        if text.get_text().isdigit():
            assert text.get_window_extent(renderer).y1 <= top + 1
    plt.close(fig)
