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
