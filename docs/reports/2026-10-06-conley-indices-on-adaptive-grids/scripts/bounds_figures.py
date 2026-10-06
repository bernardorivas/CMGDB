"""Figures for Part I, from the outputs of bounds_mechanism.py.

  figures/bounds_e1_number_line.pdf  E1 near the repeller p = 1: the cells of
      the final grid with their depths, S and A = X minus S, the hull of X
      (the bounds before the fix) and the union of the cubes (the bounds
      after it), the cubes c_k with their boxes |c_k| and the boxes
      phi(|c_k|) that the code assigned to them before the fix, the images
      under F, and the maps on the cubes.
  figures/bounds_e1_graph.pdf  E1: the graph of f over the cubes, with the
      squares |c_k| x |c_j| of the map on the cubes after the fix (left) and
      before it (right), and the graph of phi^-1 o f o phi.
  figures/bounds_e4_saddle.pdf  E4 near the saddle (0, 1): the final grid with
      S, A and F(S), and the cubes c_k (offsets (0, k)) with the boxes the
      code assigned to them before and after the fix, with the image of the
      exit cube c_2.

The rectangles drawn as "before the fix" are those the box map received on
the build given by --before (default upstream 1.5.2), and those drawn as
"after the fix" are those it received on the build given by --after
(default master 96642f2).  The script checks that they are the rectangles
that bounds_common.cube_geometry predicts.

Usage:  python bounds_figures.py [--before TAG] [--after TAG]
Needs matplotlib (no CMGDB).  Runtime: about 2 s.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Palette (validated for color-vision deficiencies with the dataviz validator):
# S and its cube blue, A and its cubes orange, the geometry before the fix violet.
COL = dict(S="#2a78d6", A="#eb6834", before="#4a3aa7", ink="#0b0b0b", muted="#52514e",
           line="#8d8b84", light="#d9d8d4", wash="#f0efec")
FILL_ALPHA = 0.30

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "hatch.linewidth": 0.6, "legend.fontsize": 7.5, "legend.frameon": False,
})


def save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"wrote {path}")


def morse_set(rec, case, point):
    for ms in rec["cases"][case]["morse_sets"]:
        if ms["fixed"] and tuple(ms["fixed"][0]) == tuple(point):
            return rec["cases"][case], ms
    raise KeyError((case, point))


def as_dict(pairs):
    return {tuple(q): v for q, v in pairs}


def recorded_boxes(ms, which):
    """The rectangle the box map received for each cube, checked against the
    prediction from the bounds before or after the fix."""
    an = ms["analysis"]
    pred = as_dict(an["rect_" + which])
    out = {}
    for ev in ms["recorded"]["evaluations"]:
        q = tuple(ev["cube"])
        assert ev["bounds"] in (which, "before and after"), ev["bounds"]
        assert max(abs(a - b) for a, b in zip(ev["rect"], pred[q])) == 0.0
        out.setdefault(q, ev["rect"])
    assert set(out) == set(pred), "not every cube was evaluated"
    return out


def sub(k):
    return "$c_{%d}$" % k


def set_text(label, M, k):
    inner = ", ".join("c_{%d}" % t[0] for t in M[(k,)])
    return "$%s(c_{%d}) = \\{%s\\}$" % (label, k, inner)


# ---------------------------------------------------------------------------
# (a) E1 on the number line
# ---------------------------------------------------------------------------

def figure_e1_number_line(after, before):
    case, ms_a = morse_set(after, "E1", (1.0,))
    _, ms_b = morse_set(before, "E1", (1.0,))
    an = ms_a["analysis"]
    grid, depths = case["grid"], case["depths"]
    S, A = set(an["S"]), set(an["A"])
    true_box = recorded_boxes(ms_a, "after")
    old_box = recorded_boxes(ms_b, "before")
    maps = {k: {tuple(q): [tuple(t) for t in v] for q, v in an["maps"][k]} for k in ("before", "after")}
    ob, tb = an["bounds_before"], an["bounds_after"]
    S_c = {tuple(q) for q in an["S_cubes"]}
    F = C.F_xs
    x0, x1 = 0.735, 1.215

    fig, ax = plt.subplots(figsize=(6.5, 4.55))
    fig.subplots_adjust(left=0.115, right=0.765, top=0.955, bottom=0.085)
    ax.set_xlim(x0, x1)
    rows = {}

    def box(xa, xb, y, h=0.56, fc="white", ec=COL["line"], lw=0.6, ls="-", hatch=None, alpha=1.0, z=2):
        ax.add_patch(Rectangle((xa, y - h / 2), xb - xa, h, facecolor=fc, edgecolor="none", alpha=alpha, zorder=z))
        if hatch:
            ax.add_patch(Rectangle((xa, y - h / 2), xb - xa, h, facecolor="none", edgecolor=fc, lw=0,
                                   hatch=hatch, alpha=0.75, zorder=z + 0.05))
        ax.add_patch(Rectangle((xa, y - h / 2), xb - xa, h, facecolor="none", edgecolor=ec, lw=lw, ls=ls,
                               zorder=z + 0.1))

    def role_color(k):
        return COL["S"] if (k,) in S_c else COL["A"]

    def bar(lo, hi, y, color, lo_c, hi_c, h=0.2):
        """Image [lo, hi] with the parts outside [lo_c, hi_c], which the cover
        clamps, drawn lighter."""
        a, b = max(lo, lo_c), min(hi, hi_c)
        if lo < a:
            ax.add_patch(Rectangle((lo, y - h / 2), min(a, hi) - lo, h, facecolor=color, alpha=0.25, lw=0, zorder=3))
        if hi > b:
            ax.add_patch(Rectangle((max(b, lo), y - h / 2), hi - max(b, lo), h, facecolor=color, alpha=0.25, lw=0, zorder=3))
        if a < b:
            ax.add_patch(Rectangle((a, y - h / 2), b - a, h, facecolor=color, alpha=0.9, lw=0, zorder=3))

    # row: F(S)
    y = 11.0
    rows[y] = "$F(S)$"
    fs = F(grid[an["S"][0]])
    ax.add_patch(Rectangle((fs[0], y - 0.1), fs[1] - fs[0], 0.2, facecolor=COL["ink"], lw=0, zorder=3))
    # row: final grid
    y = 10.1
    rows[y] = "final grid"
    for i, b in enumerate(grid):
        if b[1] < x0 or b[0] > x1:
            continue
        if i in S:
            box(b[0], b[1], y, fc=COL["S"], alpha=FILL_ALPHA, ec=COL["ink"], lw=0.7)
        elif i in A:
            box(b[0], b[1], y, fc=COL["A"], alpha=FILL_ALPHA, ec=COL["ink"], lw=0.7, hatch="////")
        else:
            box(max(b[0], x0), min(b[1], x1), y, ec=COL["line"])
        if b[0] >= x0 and b[1] <= x1:
            ax.text((b[0] + b[1]) / 2, y + 0.43, str(depths[i]), ha="center", va="bottom", fontsize=6.5,
                    color=COL["muted"])
    ax.text(x0 + 0.004, y + 0.43, "depth", ha="left", va="bottom", fontsize=6.5, color=COL["muted"])
    # rows: the two extents
    y = 9.05
    rows[y] = ""
    for (lo, hi, yy, color, ls, text) in ((tb[0], tb[1], y + 0.2, COL["ink"], "-", "union of the cubes: bounds after the fix"),
                                          (ob[0], ob[1], y - 0.2, COL["before"], "--", "hull of $X$: bounds before the fix")):
        ax.plot([lo, hi], [yy, yy], color=color, lw=1.0, ls=ls, zorder=4, solid_capstyle="butt")
        for xv in (lo, hi):
            ax.plot([xv, xv], [yy - 0.1, yy + 0.1], color=color, lw=1.0, zorder=4)
        ax.text(hi + 0.006, yy, text, ha="left", va="center", fontsize=7.5, color=color)

    # after the fix
    y = 7.95
    rows[y] = "$|c_k|$"
    for k in range(3):
        r = true_box[(k,)]
        box(r[0], r[1], y, fc=role_color(k), alpha=FILL_ALPHA, ec=COL["ink"], lw=1.0,
            hatch="////" if role_color(k) == COL["A"] else None)
        ax.text((r[0] + r[1]) / 2, y, sub(k), ha="center", va="center", fontsize=8.5, zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    img_rows_after = [7.25, 6.8, 6.35]
    for k, yy in enumerate(img_rows_after):
        rows[yy] = "$F(|c_{%d}|)$" % k
        im = F(true_box[(k,)])
        bar(im[0], im[1], yy, role_color(k), tb[0], tb[1])
        ax.annotate(set_text("\\mathcal{F}", maps["after"], k), xy=(1.0, yy), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5)
    for xv in [tb[0]] + [true_box[(k,)][1] for k in range(3)]:
        ax.plot([xv, xv], [img_rows_after[-1] - 0.25, y + 0.28], color=COL["line"], lw=0.5, zorder=1)
    ax.text(x0 + 0.004, y + 0.5, "after the fix", ha="left", va="bottom", fontsize=8, color=COL["ink"])

    # before the fix
    y = 5.15
    rows[y] = "$\\varphi(|c_k|)$"
    for k in range(3):
        r = old_box[(k,)]
        box(r[0], r[1], y, fc=role_color(k), alpha=FILL_ALPHA, ec=COL["before"], lw=1.0, ls=(0, (3, 1.5)),
            hatch="////" if role_color(k) == COL["A"] else None)
        ax.text((r[0] + r[1]) / 2, y, "$\\varphi(|c_{%d}|)$" % k, ha="center", va="center", fontsize=8, zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    img_rows_before = [4.45, 4.0, 3.55]
    for k, yy in enumerate(img_rows_before):
        rows[yy] = "$F(\\varphi(|c_{%d}|))$" % k
        im = F(old_box[(k,)])
        bar(im[0], im[1], yy, role_color(k), ob[0], ob[1])
        ax.annotate(set_text("\\tilde{\\mathcal{F}}", maps["before"], k), xy=(1.0, yy),
                    xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5,
                    color=COL["before"] if (k,) not in S_c and S_c & set(maps["before"][(k,)]) else COL["ink"])
    for xv in [ob[0]] + [old_box[(k,)][1] for k in range(3)]:
        ax.plot([xv, xv], [img_rows_before[-1] - 0.25, y + 0.28], color=COL["before"], lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.text(x0 + 0.004, y + 0.5, "before the fix", ha="left", va="bottom", fontsize=8, color=COL["before"])
    # the exit cube c_2 whose image meets the rectangle of c_1
    im2 = F(old_box[(2,)])
    lo, hi = max(im2[0], old_box[(1,)][0]), min(im2[1], old_box[(1,)][1])
    yy = img_rows_before[2]
    ax.plot([lo, hi], [yy - 0.22, yy - 0.22], color=COL["before"], lw=2.2, solid_capstyle="butt", zorder=5)
    ax.annotate("$F(\\varphi(|c_2|))$ meets $\\varphi(|c_1|)$", xy=((lo + hi) / 2, yy - 0.24), xytext=(0.86, 2.75),
                fontsize=7.5, color=COL["before"], ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=COL["before"], lw=0.6, shrinkA=1, shrinkB=1))

    # the fixed point
    ax.plot([1.0, 1.0], [2.45, 11.35], color=COL["ink"], lw=0.6, ls=(0, (1, 1.5)), zorder=7)
    ax.text(1.0, 11.4, "$p = 1$", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(2.35, 11.75)
    ax.set_yticks(list(rows))
    ax.set_yticklabels(list(rows.values()))
    ax.tick_params(axis="y", length=0)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    ax.set_xlabel("$x$", labelpad=1)
    handles = [Rectangle((0, 0), 1, 1, facecolor=COL["S"], alpha=FILL_ALPHA, edgecolor=COL["ink"], lw=0.7),
               Rectangle((0, 0), 1, 1, facecolor=(0.92, 0.41, 0.2, FILL_ALPHA), edgecolor=COL["A"], lw=0.7,
                         hatch="////")]
    ax.legend(handles, ["$S$ and its cube $c_1$", "$A = X \\setminus S$ and its cubes"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.6, borderaxespad=0.2)
    save(fig, "bounds_e1_number_line.pdf")


# ---------------------------------------------------------------------------
# (b) E1: graph of f and the maps on the cubes
# ---------------------------------------------------------------------------

def figure_e1_graph(after, before):
    import numpy as np
    from matplotlib.lines import Line2D
    _, ms_a = morse_set(after, "E1", (1.0,))
    _, ms_b = morse_set(before, "E1", (1.0,))
    an = ms_a["analysis"]
    true_box = recorded_boxes(ms_a, "after")
    recorded_boxes(ms_b, "before")
    maps = {k: {tuple(q): [tuple(t) for t in v] for q, v in an["maps"][k]} for k in ("before", "after")}
    tb = an["bounds_after"]
    ph = an["phi"][0]
    phi = lambda x: ph["origin_old"] + ph["slope"] * (x - ph["origin_true"])
    phinv = lambda y: ph["origin_true"] + (y - ph["origin_old"]) / ph["slope"]
    f = C.f_xs
    lim = (0.835, 1.118)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.95))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.255, wspace=0.22)
    xs = np.linspace(lim[0], lim[1], 400)
    edges = [tb[0]] + [true_box[(k,)][1] for k in range(3)]
    S_c = {tuple(q) for q in an["S_cubes"]}
    for ax, label, M, title in ((axes[0], "after", maps["after"], "after the fix: $\\mathcal{F}$"),
                                (axes[1], "before", maps["before"], "before the fix: $\\tilde{\\mathcal{F}}$")):
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        # colored bands along the axes: the cubes and their roles
        for k in range(3):
            r = true_box[(k,)]
            col = COL["S"] if (k,) in S_c else COL["A"]
            ax.add_patch(Rectangle((r[0], lim[0]), r[1] - r[0], 0.008, facecolor=col, alpha=0.8, lw=0, zorder=2))
            ax.add_patch(Rectangle((lim[0], r[0]), 0.008, r[1] - r[0], facecolor=col, alpha=0.8, lw=0, zorder=2))
            ax.text((r[0] + r[1]) / 2, lim[0] + 0.012, "$c_{%d}$" % k, ha="center", va="bottom", fontsize=8)
            ax.text(lim[0] + 0.012, (r[0] + r[1]) / 2, "$c_{%d}$" % k, ha="left", va="center", fontsize=8)
        # squares c_k x c_j with c_j in M(c_k)
        for k in range(3):
            for (j,) in M[(k,)]:
                rk, rj = true_box[(k,)], true_box[(j,)]
                bad = (k,) not in S_c and (j,) in S_c
                ax.add_patch(Rectangle((rk[0], rj[0]), rk[1] - rk[0], rj[1] - rj[0],
                                       facecolor=COL["before"] if bad else COL["light"], alpha=0.3 if bad else 0.8,
                                       edgecolor="none", zorder=1))
                if bad:
                    ax.add_patch(Rectangle((rk[0], rj[0]), rk[1] - rk[0], rj[1] - rj[0], facecolor="none",
                                           edgecolor=COL["before"], lw=1.0, hatch="xxx", zorder=1.2))
        for e in edges:
            ax.plot([e, e], [tb[0], tb[1]], color=COL["muted"], lw=0.4, zorder=1.5)
            ax.plot([tb[0], tb[1]], [e, e], color=COL["muted"], lw=0.4, zorder=1.5)
        ax.plot(lim, lim, color=COL["line"], lw=0.6, zorder=2)
        ax.plot(xs, [f(t) for t in xs], color=COL["ink"], lw=1.3, zorder=4)
        ax.plot([1.0], [1.0], "o", ms=3.5, color=COL["ink"], zorder=5)
        ax.annotate("$p = 1$", xy=(1.0, 1.0), xytext=(0.955, 1.075), fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color=COL["ink"], lw=0.6, shrinkA=1, shrinkB=2))
        if label == "before":
            ax.plot(xs, [phinv(f(phi(t))) for t in xs], color=COL["before"], lw=1.3, ls=(0, (4, 2)), zorder=4)
            q = phinv(1.0)
            ax.plot([q], [q], "o", ms=3.5, color=COL["before"], zorder=5)
            ax.text(1.078, 1.03, "$\\varphi^{-1}(1)$", fontsize=7.5, color=COL["before"], ha="center", va="center",
                    zorder=6)
            k2, k1 = true_box[(2,)], true_box[(1,)]
            ax.annotate("$c_1 \\in \\tilde{\\mathcal{F}}(c_2)$", xy=((k2[0] + k2[1]) / 2, (k1[0] + k1[1]) / 2),
                        xytext=(1.075, 0.895), fontsize=7.5, color=COL["before"], ha="center",
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"),
                        arrowprops=dict(arrowstyle="-", color=COL["before"], lw=0.6, shrinkA=1, shrinkB=1))
        ticks = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("$x$", labelpad=1)
        ax.set_title(title, fontsize=9, color=COL["before"] if label == "before" else COL["ink"], pad=4)
    handles = [Line2D([], [], color=COL["ink"], lw=1.3),
               Line2D([], [], color=COL["before"], lw=1.3, ls=(0, (4, 2))),
               Line2D([], [], color=COL["line"], lw=0.6),
               Rectangle((0, 0), 1, 1, facecolor=COL["light"], edgecolor="none"),
               Rectangle((0, 0), 1, 1, facecolor=(0.29, 0.23, 0.65, 0.3), edgecolor=COL["before"], hatch="xxx", lw=1.0)]
    fig.legend(handles, ["graph of $f$", "graph of $\\varphi^{-1}\\circ f\\circ\\varphi$", "diagonal",
                         "$|c_k| \\times |c_j|$ with $c_j$ in the image of $c_k$",
                         "the same, with $c_k \\in A$\u2032 and $c_j = c_1$"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=2.0)
    save(fig, "bounds_e1_graph.pdf")


# ---------------------------------------------------------------------------
# (c) E4 near the saddle (0, 1)
# ---------------------------------------------------------------------------

def figure_e4_saddle(after, before):
    from matplotlib.lines import Line2D
    case, ms_a = morse_set(after, "E4", (0.0, 1.0))
    _, ms_b = morse_set(before, "E4", (0.0, 1.0))
    an = ms_a["analysis"]
    grid, depths = case["grid"], case["depths"]
    S, A = set(an["S"]), set(an["A"])
    true_box = recorded_boxes(ms_a, "after")
    old_box = recorded_boxes(ms_b, "before")
    assert sorted(true_box) == [(0, 0), (0, 1), (0, 2)]
    ob, tb = an["bounds_before"], an["bounds_after"]
    F = C.F_xs
    fs = F(grid[an["S"][0]])
    xw, yw = (-0.006, 0.0835), (0.845, 1.105)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4.6), sharey=True, gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.185, wspace=0.08)
    S_c = {tuple(q) for q in an["S_cubes"]}
    name = lambda q: "c_{%d}" % q[1]

    def fill(ax, r, role, z=1, alpha=FILL_ALPHA):
        col = COL["S"] if role == "S" else COL["A"]
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor=col, alpha=alpha, lw=0, zorder=z))
        if role == "A":
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor="none", edgecolor=col, lw=0,
                                   hatch="////", alpha=0.6, zorder=z + 0.1))

    def outline(ax, r, color, lw=1.0, ls="-", z=4):
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor="none", edgecolor=color,
                               lw=lw, ls=ls, zorder=z))

    # panel 1: the final grid in the column of S
    ax = axes[0]
    for i, b in enumerate(grid):
        if b[2] <= xw[0] or b[0] >= xw[1] or b[3] <= yw[0] or b[1] >= yw[1]:
            continue
        if i in S:
            fill(ax, b, "S")
        elif i in A:
            fill(ax, b, "A")
        outline(ax, b, COL["line"], lw=0.5, z=2)
        cx, cy = (b[0] + min(b[2], xw[1])) / 2, (max(b[1], yw[0]) + min(b[3], yw[1])) / 2
        if i in S:
            cx = b[0] + 0.75 * (b[2] - b[0])
        ax.text(cx, cy, str(depths[i]), ha="center", va="center", fontsize=5.5 if depths[i] >= 13 else 6.5,
                color=COL["muted"], zorder=6,
                bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.85))
    outline(ax, fs, COL["ink"], lw=1.0, ls=(0, (1, 1.2)), z=5)
    outline(ax, [tb[0], tb[1], tb[2], tb[3]], COL["ink"], lw=1.3, z=4.5)
    outline(ax, [ob[0], ob[1], ob[2], ob[3]], COL["before"], lw=1.3, ls=(0, (3, 1.5)), z=4.6)
    ax.set_title("final grid near $S$", fontsize=9, pad=4)

    # panels 2 and 3: the cubes with their rectangles after and before the fix
    def cube_panel(ax, boxes, color, ls, fmt, box_fmt, bounds, title):
        """fmt labels a cube, box_fmt the box on which F was evaluated."""
        for q, r in sorted(boxes.items()):
            fill(ax, r, "S" if q in S_c else "A")
            outline(ax, r, color, lw=1.2, ls=ls, z=4)
            ax.text(0.0765, (r[1] + r[3]) / 2, "$" + fmt % name(q) + "$", fontsize=8, ha="right", va="center", zorder=8,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
        im = F(boxes[(0, 2)])
        lo, hi = max(im[1], bounds[1]), min(im[3], bounds[3])
        ax.add_patch(Rectangle((im[0], im[1]), im[2] - im[0], im[3] - im[1], facecolor="none",
                               edgecolor=COL["ink"], lw=0.9, ls=(0, (2, 1)), zorder=5))
        ax.add_patch(Rectangle((im[0], lo), im[2] - im[0], hi - lo, facecolor=COL["ink"], alpha=0.12, lw=0, zorder=5))
        ax.text(im[2] + 0.002, yw[1] - 0.004, "$F(%s)$" % (box_fmt % name((0, 2))), fontsize=7.5, ha="left", va="top",
                zorder=8)
        ax.set_title(title, fontsize=9, pad=4, color=color)
        return im

    cube_panel(axes[1], true_box, COL["ink"], "-", "%s", "|%s|", tb, "after the fix")
    im = cube_panel(axes[2], old_box, COL["before"], (0, (3, 1.5)), "\\varphi(|%s|)", "\\varphi(|%s|)", ob,
                    "before the fix")
    # the image of the rectangle of the exit cube c_2 meets the rectangle of c_1
    r1 = old_box[(0, 1)]
    meet = [im[0], max(im[1], r1[1]), im[2], min(im[3], r1[3])]
    axes[2].add_patch(Rectangle((meet[0], meet[1]), meet[2] - meet[0], meet[3] - meet[1], facecolor=COL["before"],
                                alpha=0.75, lw=0, zorder=6))
    axes[2].annotate("$F(\\varphi(|c_2|))\\cap\\varphi(|c_1|)$", xy=((meet[0] + meet[2]) / 2, meet[1]),
                     xytext=(0.038, 0.94), fontsize=7.5, color=COL["before"], ha="center", va="top", zorder=8,
                     bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"),
                     arrowprops=dict(arrowstyle="-", color=COL["before"], lw=0.6, shrinkA=1, shrinkB=1))
    for ax in axes:
        ax.set_xlim(*xw)
        ax.set_ylim(*yw)
        ax.set_xticks([0.0, 0.04, 0.08])
        ax.set_xlabel("$x$", labelpad=1)
        ax.plot([0.0], [1.0], "o", ms=3.5, color=COL["ink"], zorder=9, clip_on=False)
        ax.axvspan(xw[0], 0.0, facecolor=COL["wash"], lw=0, zorder=0)
    axes[0].set_ylabel("$y$", labelpad=1)
    axes[0].annotate("$(0, 1)$", xy=(0.0, 1.0), xytext=(0.006, 1.004), fontsize=7.5, ha="left", va="bottom",
                     zorder=9)
    handles = [Rectangle((0, 0), 1, 1, facecolor=COL["S"], alpha=FILL_ALPHA, lw=0),
               Rectangle((0, 0), 1, 1, facecolor=(0.92, 0.41, 0.2, FILL_ALPHA), edgecolor=COL["A"], hatch="////", lw=0),
               Line2D([], [], color=COL["ink"], lw=1.0, ls=(0, (1, 1.2))),
               Line2D([], [], color=COL["ink"], lw=1.3),
               Line2D([], [], color=COL["before"], lw=1.3, ls=(0, (3, 1.5))),
               Line2D([], [], color=COL["ink"], lw=0.9, ls=(0, (2, 1))),
               Rectangle((0, 0), 1, 1, facecolor=COL["wash"], lw=0)]
    fig.legend(handles, ["$S$ and its cube $c_1$", "$A = X\\setminus S$ and its cubes", "$F(S)$",
                         "union of the cubes, and $|c_k|$", "hull of $X$, and $\\varphi(|c_k|)$", "image under $F$",
                         "outside the phase space"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)
    save(fig, "bounds_e4_saddle.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--before", default="cmgdb-1.5.2")
    ap.add_argument("--after", default="cmgdb-1.5.3_fork.7.dev0-96642f2")
    args = ap.parse_args()
    after = C.load_json("bounds_mechanism", args.after)
    before = C.load_json("bounds_mechanism", args.before)
    figure_e1_number_line(after, before)
    figure_e1_graph(after, before)
    figure_e4_saddle(after, before)


if __name__ == "__main__":
    main()
