"""Regression tests for the Morse set plots (review findings C01, C02 and C04-C22)."""

import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import CMGDB
from CMGDB.PlotMorseSets import _exposed_faces

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


def speck_rows():
    """A 5x5 block (set 0) and one 0.01 box at 0.70 (set 1)."""
    rows = [[i * H, j * H, (i + 1) * H, (j + 1) * H, 0] for i in range(5) for j in range(5)]
    return rows + [[0.70, 0.70, 0.71, 0.71, 1]]


def test_zoom_inset_draws_the_sets_at_their_true_size():
    # C09: the inset window came from the true extents while the inset drew
    # the sets inflated, so a set inflated 4x filled the inset solid. The
    # inset is what shows a set at its true size; scale_factor applies to the
    # main axes only.
    fig, ax = CMGDB.PlotMorseSets(speck_rows(), scale_factor=[1, 4], zoom_nodes=[1],
                                  show=False)
    inset = ax.child_axes[0]
    speck = inset.patches[1].get_path().vertices
    assert np.allclose(speck.min(axis=0), [0.70, 0.70])
    assert np.allclose(speck.max(axis=0), [0.71, 0.71])
    (x0, x1), (y0, y1) = inset.get_xlim(), inset.get_ylim()
    assert x0 < 0.70 and 0.71 < x1 and y0 < 0.70 and 0.71 < y1
    main = ax.patches[1].get_path().vertices
    assert np.allclose(main.min(axis=0), [0.685, 0.685])
    plt.close(fig)


@pytest.mark.parametrize("fig_w, fig_h, zoom_pos", [(8, 8, None), (12, 4, None),
                                                    (8, 8, [0.05, 0.5, 0.5, 0.25])])
def test_square_zoom_magnifies_both_axes_alike(fig_w, fig_h, zoom_pos):
    # C18: the window was squared in data units. On x in [0, 100], y in [0, 1]
    # its y side spanned more than the main axes, and a square box came out
    # of the inset as a 157 x 1.6 px sliver.
    rows = [[i, j / 100, i + 1, (j + 1) / 100, 0] for i in range(0, 100, 10)
            for j in range(0, 100, 10)]
    rows += [[50, 0.50, 51, 0.51, 1]]
    fig, ax = CMGDB.PlotMorseSets(rows, zoom_nodes=[1], zoom_pos=zoom_pos,
                                  fig_w=fig_w, fig_h=fig_h, show=False)
    fig.canvas.draw()
    inset = ax.child_axes[0]

    def size(axes):
        (x0, y0), (x1, y1) = axes.transData.transform([(50, 0.50), (51, 0.51)])
        return x1 - x0, y1 - y0

    (main_w, main_h), (inset_w, inset_h) = size(ax), size(inset)
    assert inset_w / main_w == pytest.approx(inset_h / main_h, rel=0.01)
    assert inset_w / main_w > 5
    plt.close(fig)


def render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)


def pixel(image, ax, x, y, dx=0):
    px, py = ax.transData.transform((x, y))
    return image[int(image.shape[0] - py), int(px) + dx]


@pytest.mark.parametrize("kwargs", [
    {'alpha': 0.5, 'merge_boxes': True},
    {'alpha': 0.5, 'merge_boxes': False},
    {'clist': ['#1f77b480'], 'merge_boxes': False},       # alpha carried by the color
])
def test_translucent_sets_do_not_darken_along_their_edges(kwargs):
    # C10: the face-colored edge was stroked over a translucent fill, and
    # matplotlib applies alpha to each stroke, not to the set as a whole: drawn
    # box by box, every seam came out darker than the interior, and the merged
    # outline darkened the rim of the set.
    rows = [[0, 0, 1, 1, 0], [1, 0, 2, 1, 0]]
    fig, ax = CMGDB.PlotMorseSets(rows, linewidth=2, show=False, **kwargs)
    image = render(fig)
    interior = pixel(image, ax, 0.5, 0.5)
    assert np.abs(pixel(image, ax, 1.0, 0.5) - interior).max() <= 2       # the seam
    assert np.abs(pixel(image, ax, 0.0, 0.5, dx=1) - interior).max() <= 2  # inside the rim
    plt.close(fig)


def stacked_rows():
    """3-D boxes over a 10x10 grid in (x, y), one deep for x < 0.5, six deep beyond."""
    return [[i * H, j * H, k * H, (i + 1) * H, (j + 1) * H, (k + 1) * H, 0]
            for i in range(10) for j in range(10) for k in range(1 if i < 5 else 6)]


@pytest.mark.parametrize("kwargs", [{'alpha': 0.3}, {'clist': ['#1f77b44d']}])
def test_translucent_projection_shows_where_boxes_stack(kwargs):
    # C19: a translucent set projected from 3-D was merged into one flat
    # outline, while drawn box by box it is darker where boxes stack, as the
    # merge_boxes docstring says the two pictures are the same.
    fig, ax = CMGDB.PlotMorseSets(stacked_rows(), show=False, **kwargs)
    fig_b, ax_b = CMGDB.PlotMorseSets(stacked_rows(), merge_boxes=False, show=False, **kwargs)
    image, per_box = render(fig), render(fig_b)
    assert (pixel(image, ax, 0.75, 0.55) < pixel(image, ax, 0.25, 0.55)).all()
    assert (np.abs(image - per_box).max(axis=2) > 8).mean() < 0.005
    plt.close(fig)
    plt.close(fig_b)
    # An opaque projection still merges: there is nothing to accumulate.
    fig, ax = CMGDB.PlotMorseSets(stacked_rows(), show=False)
    assert len(ax.patches) == 1 and len(ax.collections) == 0
    plt.close(fig)


def test_translucent_color_keeps_inflated_overlaps():
    # C19: only a scalar alpha counted as translucent, so a color carrying its
    # own alpha merged its inflated boxes and lost the overlaps.
    rows = [[i * H, 0, (i + 1) * H, H, 0] for i in range(0, 10, 2)]
    fig, ax = CMGDB.PlotMorseSets(rows, clist=['#1f77b480'], scale_factor=[3], show=False)
    assert len(ax.patches) == 0 and len(ax.collections) == 1
    image = render(fig)
    overlap, single = pixel(image, ax, 0.15, 0.05), pixel(image, ax, 0.25, 0.05)
    assert (overlap < single).all()
    plt.close(fig)


def test_translucent_unscaled_set_still_merges():
    # C19: only sets whose boxes overlap leave the merged path. Here the
    # inflated set 0 does, and set 1, unscaled and with no cell repeated,
    # does not.
    rows = [[i * H, j * H, (i + 1) * H, (j + 1) * H, 0] for i in range(3) for j in range(3)]
    rows += [[i * H, j * H, (i + 1) * H, (j + 1) * H, 1] for i in range(5, 8) for j in range(3)]
    fig, ax = CMGDB.PlotMorseSets(rows, scale_factor=[1.5, 1], alpha=0.5, show=False)
    assert len(ax.patches) == 1 and len(ax.collections) == 1
    plt.close(fig)


def cube_rows(n=3):
    """An n x n x n block of 0.1 cubes, all in set 0."""
    return [[i * H, j * H, k * H, (i + 1) * H, (j + 1) * H, (k + 1) * H, 0]
            for i in range(n) for j in range(n) for k in range(n)]


def embedded_ppi(path):
    """Pixels per inch of the page of the one bitmap in an uncompressed PDF."""
    raw = path.read_bytes()
    # The bitmap and its alpha mask each state the width.
    widths = {int(w) for w in re.findall(rb'/Width\s+(\d+)', raw)}
    placed = re.findall(rb'([-\d.]+) [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ cm\s*/\w+ Do', raw)
    assert len(widths) == 1 and len(placed) == 1
    return widths.pop() / (float(placed[0]) / 72.0)


def png_dpi(path):
    return Image.open(path).info['dpi'][0]


@pytest.mark.parametrize("plot, rows", [(CMGDB.PlotMorseSets, corner_rows()),
                                        (CMGDB.PlotMorseSets3D, cube_rows())])
def test_rasterized_saves_have_the_requested_dpi(monkeypatch, tmp_path, plot, rows):
    # C02: a rasterized save multiplied dpi by a probe of how much of the
    # page the bitmap covers, so dpi=100 embedded 119 px per inch (2-D) and
    # 132 (3-D) in a PDF and enlarged a PNG alike. A bitmap covering part of
    # the page already has dpi pixels per inch of it.
    monkeypatch.setitem(matplotlib.rcParams, 'pdf.compression', 0)
    fig, ax = plot(rows, rasterize=True, dpi=100, fig_fname=str(tmp_path / 'r.pdf'),
                   show=False)
    assert embedded_ppi(tmp_path / 'r.pdf') == pytest.approx(100, rel=0.01)
    plt.close(fig)
    fig, ax = plot(rows, rasterize=True, dpi=100, fig_fname=str(tmp_path / 'r.png'),
                   show=False)
    assert png_dpi(tmp_path / 'r.png') == pytest.approx(100, rel=0.01)
    plt.close(fig)


@pytest.mark.parametrize("plot, rows", [
    (CMGDB.PlotMorseSets, corner_rows()),
    (CMGDB.PlotMorseSets, line_rows()),
    (CMGDB.PlotMorseSets1D, line_rows()),
    (CMGDB.PlotMorseSets3D, cube_rows()),
    (CMGDB.PlotMorseSetsScatter, corner_rows()),
])
@pytest.mark.parametrize("rasterize", [False, True])
def test_dpi_figure_saves_at_the_figure_dpi(tmp_path, plot, rows, rasterize):
    # C22: savefig takes dpi='figure', and 1.3.2 passed it on, but the dpi was
    # multiplied by a scale before saving, so 'figure' * 1.0 raised TypeError.
    out = tmp_path / 'f.png'
    fig, ax = plot(rows, fig_fname=str(out), dpi='figure', rasterize=rasterize, show=False)
    assert png_dpi(out) == pytest.approx(fig.dpi, rel=0.01)
    plt.close(fig)


def test_default_dpi_rises_only_for_rasterized_vector_files(monkeypatch, tmp_path):
    # C17: dpi=None became 600 whenever the plot was rasterized, whatever the
    # format, so a PNG of a 3-D plot doubled its resolution once the face
    # count crossed RASTERIZE_FACES. The 600 is for the bitmap inside a vector
    # page; a bitmap format keeps 300.
    import importlib
    plot_module = importlib.import_module('CMGDB.PlotMorseSets')
    monkeypatch.setattr(plot_module, 'RASTERIZE_FACES', 40)
    monkeypatch.setitem(matplotlib.rcParams, 'pdf.compression', 0)
    fig, ax = CMGDB.PlotMorseSets3D(cube_rows(), fig_fname=str(tmp_path / 'auto.png'),
                                    show=False)
    assert ax.collections[0].get_rasterized()
    assert png_dpi(tmp_path / 'auto.png') == pytest.approx(300, rel=0.01)
    plt.close(fig)
    # A path, an upper-case extension, and no extension (savefig.format, png).
    for name, saved in (('a.png', 'a.png'), ('b.JPG', 'b.JPG'), ('c', 'c.png')):
        fig, ax = CMGDB.PlotMorseSets(corner_rows(), rasterize=True,
                                      fig_fname=tmp_path / name, show=False)
        assert png_dpi(tmp_path / saved) == pytest.approx(300, rel=0.01)
        plt.close(fig)
    fig, ax = CMGDB.PlotMorseSets(corner_rows(), rasterize=True,
                                  fig_fname=str(tmp_path / 'flat.pdf'), show=False)
    assert embedded_ppi(tmp_path / 'flat.pdf') == pytest.approx(600, rel=0.01)
    plt.close(fig)


@pytest.mark.parametrize("plot, num_morse_sets", [(CMGDB.PlotMorseSets, []),
                                                  (CMGDB.PlotMorseSetsScatter, []),
                                                  (CMGDB.PlotBoxesScatter, [None])])
def test_positional_arguments_bind_as_in_1_3_2(tmp_path, plot, num_morse_sets):
    # C16: margin was inserted after ylim, and edge_clr and the zoom options
    # before fig_fname, so every 1.3.2 positional argument from axis_labels
    # on bound to the parameter before it: axis_labels=False became margin=0
    # and the labels stayed, and a full call failed on the file name taken
    # for a font size. The 1.3.2 order is (..., xlim, ylim, axis_labels,
    # xlabel, ylabel, fontsize, fig_fname, dpi); the new options follow it.
    lead = [corner_rows()] + num_morse_sets + [None, None, None, None, None, 8, 8, None, None]
    fig, ax = plot(*lead, False, show=False)
    assert ax.get_xlabel() == '' and ax.get_ylabel() == ''
    plt.close(fig)
    out = tmp_path / 'positional.png'
    fig, ax = plot(*lead, True, 'u', 'v', 12, str(out), 50, show=False)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('u', 'v')
    assert ax.xaxis.label.get_fontsize() == 12
    assert png_dpi(out) == pytest.approx(50, rel=0.01)
    plt.close(fig)
    with pytest.raises(TypeError):                # the new options take keywords only
        plot(*lead, True, 'u', 'v', 12, None, 50, 0.1, show=False)


def x_planes(faces):
    """The x coordinates of the faces normal to x."""
    return sorted({round(float(f[0, 0]), 9) for f in faces if np.ptp(f[:, 0]) == 0})


def test_shrunk_3d_boxes_keep_every_face():
    # C12: faces shared with a same-set neighbor were culled before the
    # boxes were scaled. Shrunk boxes come apart, so the culled faces left
    # each one open toward its neighbors and the inner boxes of a block
    # with no face at all.
    pair = [[0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 2, 1, 1, 0]]
    faces, _, _ = _exposed_faces(pair, [0], [0.5])
    assert len(faces) == 12
    assert x_planes(faces) == [0.25, 0.75, 1.25, 1.75]
    assert len(_exposed_faces(cube_rows(), [0], [0.5])[0]) == 6 * 27
    # Enlarged boxes still overlap, so their shared faces stay hidden.
    for factor in (1, 2):
        faces, _, _ = _exposed_faces(pair, [0], [factor])
        assert len(faces) == 10
        assert len(_exposed_faces(cube_rows(), [0], [factor])[0]) == 6 * 9


def test_unaligned_3d_set_falls_back_alone():
    # C15: one grid was fitted to the boxes of every set, drawn or not, so a
    # single box of another size or offset anywhere turned culling off for
    # all sets: their interior faces were drawn too, six per box, and showed
    # through the antialiasing seams as a dark grid. Neighbors are only
    # looked for within a set, so each set needs a grid of its own.
    for odd in ([[2.0, 2.0, 2.0, 2.07, 2.07, 2.07, 1]],                  # another size
                [[2.03, 2.0, 2.0, 2.13, 2.1, 2.1, 1],                    # off the lattice
                 [2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 1]]):
        faces, _, _ = _exposed_faces(cube_rows() + odd, [0], [1, 1])
        assert len(faces) == 6 * 9
        faces, labels, _ = _exposed_faces(cube_rows() + odd, [0, 1], [1, 1])
        assert np.sum(labels == 0) == 6 * 9 and np.sum(labels == 1) == 6 * len(odd)


def test_3d_limits_contain_the_inflated_sets():
    # C14: the 3-D limits came from the raw box corners, so a set inflated
    # at the edge of the data was drawn outside the axes, over the tick
    # numbers; mplot3d does not clip it.
    w = 0.05
    rows = [[i * w, j * w, k * w, (i + 1) * w, (j + 1) * w, (k + 1) * w, 0]
            for i in range(3) for j in range(3) for k in range(3)]
    rows.append([1.0, 1.0, 1.0, 1.0 + w, 1.0 + w, 1.0 + w, 1])
    fig, ax = CMGDB.PlotMorseSets3D(rows, scale_factor=[1, 6], show=False)
    faces, _, _ = _exposed_faces(rows, [0, 1], [1, 6])
    low, high = faces.reshape(-1, 3).min(axis=0), faces.reshape(-1, 3).max(axis=0)
    assert np.allclose(high, 1.175)               # set 1 is drawn on [0.875, 1.175]^3
    for d, limits in enumerate((ax.get_xlim(), ax.get_ylim(), ax.get_zlim())):
        assert limits[0] < low[d] and high[d] < limits[1]
    plt.close(fig)
    fig, ax = CMGDB.PlotMorseSets3D(rows, scale_factor=[1, 6], xlim=[0, 1], show=False)
    assert ax.get_xlim() == pytest.approx((0, 1))  # explicit limits are kept
    plt.close(fig)


def zlabel_and_ticks(fig, ax, text):
    """The drawn z label and the z tick numbers it has to clear, in pixels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    label = [t for t in ax.texts if t.get_text() == text][0]
    low, high = sorted(ax.zaxis.get_view_interval())
    ticks = [tick.label1.get_window_extent(renderer) for tick in ax.zaxis.get_major_ticks()
             if low <= tick.get_loc() <= high and tick.label1.get_visible()
             and tick.label1.get_text()]
    return label.get_window_extent(renderer), ticks


def test_3d_figures_pickle_and_copy_with_their_z_label():
    # C11: the z label measured itself through a draw closure stored on the
    # Text, which pickle cannot reach, and which a deep copy shared: the copy
    # drew the original figure's label instead of its own.
    import copy
    import io
    import pickle
    fig, ax = CMGDB.PlotMorseSets3D(cube_rows(), zlabel='ORIG', show=False)
    clone = pickle.loads(pickle.dumps(fig))
    own, ticks = zlabel_and_ticks(clone, clone.axes[0], 'ORIG')
    assert ticks and own.x0 > max(box.x1 for box in ticks)   # still measured into place
    plt.close(clone)
    copied = copy.deepcopy(fig)
    [label] = [t for t in copied.axes[0].texts if t.get_text() == 'ORIG']
    label.set_text('COPY')
    buffer = io.StringIO()
    with matplotlib.rc_context({'svg.fonttype': 'none'}):
        copied.savefig(buffer, format='svg')
    assert 'COPY' in buffer.getvalue() and 'ORIG' not in buffer.getvalue()
    plt.close(copied)
    plt.close(fig)


@pytest.mark.parametrize("azim, side", [(-55, 'right'), (30, 'left'), (45, 'left'),
                                        (-135, 'left'), (135, 'right')])
def test_zlabel_goes_beyond_the_z_tick_numbers_on_either_side(azim, side):
    # C13: the label was always moved to the right of the z tick numbers.
    # For about half the camera azimuths mplot3d draws the z axis and its
    # numbers on the left, and the label then sat between the numbers and
    # the axis line, over the drawn sets.
    fig, ax = CMGDB.PlotMorseSets3D(cube_rows(4), azim=azim, show=False)
    own, ticks = zlabel_and_ticks(fig, ax, '$z$')
    assert ticks and not any(own.overlaps(box) for box in ticks)
    if side == 'left':
        assert own.x1 < min(box.x0 for box in ticks)
    else:
        assert own.x0 > max(box.x1 for box in ticks)
    span = (min(box.y0 for box in ticks), max(box.y1 for box in ticks))
    assert 0.5 * (own.y0 + own.y1) == pytest.approx(0.5 * sum(span), abs=1.0)
    plt.close(fig)


def test_zlabel_follows_the_z_axis_to_the_left():
    # C13: a camera turned after the plot is built moves the label too.
    fig, ax = CMGDB.PlotMorseSets3D(cube_rows(4), show=False)
    ax.view_init(elev=22, azim=45)
    own, ticks = zlabel_and_ticks(fig, ax, '$z$')
    assert own.x1 < min(box.x0 for box in ticks)
    plt.close(fig)
