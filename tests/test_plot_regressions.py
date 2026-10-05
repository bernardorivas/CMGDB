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
