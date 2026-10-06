"""Plot style shared by the Leslie figures (Part II).

Figures are drawn at a text width of 6.5 in with 8-9 pt type.  The colors
follow the invariant set, never the run:

  blue     p*, Gamma and the attractor K around p*, and the basin of Gamma
  orange   the attracting 3-cycle a3 and its basin
  aqua     the saddle 3-cycle s3
  violet   a Morse set that contains s3, p* and Gamma
  gray     the origin (dark) and Morse sets that contain none of these (light)

Any two of the first four colors differ by at least 9.2 (OKLab distance x 100)
under simulated protanopia and deuteranopia (Machado, Oliveira and Fernandes,
severity 1) and by at least 16.3 in normal vision.  Markers differ in shape as
well as color: 0 is a square, p* a star, a3 circles and s3 triangles.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import leslie_common as LC

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
MUTED = "#52514e"
DARK_GRAY = "#3d3c39"
LIGHT_GRAY = "#a9a8a1"
PALE = "#ecebe7"
HATCH = "#9a9993"
TEXT_WIDTH = 6.5


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "legend.frameon": False, "hatch.linewidth": 0.5,
    })


def domain_axes(ax, labels=True):
    ax.set_xlim(LC.LOWER[0], LC.UPPER[0])
    ax.set_ylim(LC.LOWER[1], LC.UPPER[1])
    ax.set_aspect("equal")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    if labels:
        ax.set_xlabel("$x$", labelpad=1)
        ax.set_ylabel("$y$", labelpad=1)


def boxes_collection(boxes, **kw):
    polys = [[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])] for b in boxes]
    return PolyCollection(polys, **kw)


def add_boxes(ax, boxes, **kw):
    if len(boxes):
        ax.add_collection(boxes_collection(boxes, **kw))


def load_dynamics():
    return LC.load_json("leslie_dynamics")


def gamma_curve(dyn):
    """Gamma as a closed polyline (points sorted by angle about p*)."""
    G = dyn["gamma"]["curve"]
    return [p[0] for p in G] + [G[0][0]], [p[1] for p in G] + [G[0][1]]


MARK = {
    "p*": dict(marker="*", ms=9, mfc="white", mec=INK, mew=0.7, ls="none"),
    "0": dict(marker="s", ms=4.0, mfc=INK, mec=INK, mew=0.5, ls="none"),
    "a3": dict(marker="o", ms=4.5, mfc=ORANGE, mec=INK, mew=0.6, ls="none"),
    "s3": dict(marker="^", ms=5.0, mfc=AQUA, mec=INK, mew=0.6, ls="none"),
}
GAMMA_LINE = dict(color=INK, lw=0.9)


def plot_invariant_sets(ax, dyn, zorder=10, gamma=True):
    if gamma:
        gx, gy = gamma_curve(dyn)
        ax.plot(gx, gy, zorder=zorder, **GAMMA_LINE)
    ax.plot(*LC.P_STAR, zorder=zorder + 1, **MARK["p*"])
    ax.plot(0.0, 0.0, zorder=zorder + 1, clip_on=False, **MARK["0"])
    for name, pts in (("a3", LC.A3), ("s3", LC.S3)):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], zorder=zorder + 1, clip_on=False, **MARK[name])


def invariant_set_handles():
    return [Line2D([], [], label=r"$\Gamma$", **GAMMA_LINE),
            Line2D([], [], label=r"$p^{\ast}$", **MARK["p*"]),
            Line2D([], [], label=r"$0$", **MARK["0"]),
            Line2D([], [], label=r"$a_3$", **MARK["a3"]),
            Line2D([], [], label=r"$s_3$", **MARK["s3"])]


def patch(label, face, edge=None, hatch=None, alpha=1.0, lw=0.6):
    return Patch(facecolor=face, edgecolor=edge if edge else face, hatch=hatch, alpha=alpha, lw=lw, label=label)


def panel_label(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
