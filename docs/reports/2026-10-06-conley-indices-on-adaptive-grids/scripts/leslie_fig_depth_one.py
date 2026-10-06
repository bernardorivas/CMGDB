"""Figure: the cells L and C of depth 1, the box Phi(L) and the hull of f(L).

The box Phi(L) is spanned by f at the four corners of L (black dots) and
stays in x <= 9.80, but f(L) (f at 8000 uniformly distributed random points
of L, seed 0, gray) reaches x = 87.12, and f(10, 0) = (72.10, 7.00) lies in
C.  The box Phi(C), which is the hull of f(C), lies in L.

Reads outputs/leslie_depth_one_<tag>.json and writes figures/leslie_depth_one.pdf.

usage: python leslie_fig_depth_one.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
import leslie_plotstyle as S
from leslie_plotstyle import plt, Line2D


def rect(ax, b, **kw):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], **kw))


def main():
    args = LC.parse_args(__doc__)
    S.setup()
    d = LC.load_json("leslie_depth_one", args.tag)
    L, C = d["L"], d["C"]
    img = d["image_of_L"]
    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH, 3.9))
    S.domain_axes(ax)
    ax.axvline(C[0], color=S.MUTED, lw=0.8, zorder=2)
    ax.text(C[0] / 2, 66.5, "$L$", ha="center", va="top", fontsize=11)
    ax.text((C[0] + C[2]) / 2, 66.5, "$C$", ha="center", va="top", fontsize=11)

    rng = np.random.default_rng(0)
    Z = rng.uniform([L[0], L[1]], [L[2], L[3]], size=(8000, 2))
    P = LC.f_array(Z)
    ax.plot(P[:, 0], P[:, 1], ls="none", marker=".", ms=1.6, mew=0, color=S.LIGHT_GRAY, zorder=1)

    h = img["hull"]
    rect(ax, h, fill=False, ec=S.INK, lw=1.0, ls=(0, (5, 3)), zorder=4)
    c = img["corner"]
    rect(ax, c, fc=S.INK, alpha=0.18, ec="none", zorder=3)
    rect(ax, c, fill=False, ec=S.INK, lw=1.0, zorder=4)
    cc = d["image_of_C"]["corner"]
    rect(ax, cc, fill=False, ec=S.INK, lw=1.0, ls=(0, (1, 1.5)), zorder=4)
    ci = np.array(img["corner_images"])
    ax.plot(ci[:, 0], ci[:, 1], ls="none", marker="o", ms=3.2, mfc=S.INK, mec="white", mew=0.4, zorder=6,
            clip_on=False)

    p, fp = (10.0, 0.0), img["f(10,0)"]
    ax.annotate("", xy=fp, xytext=p, arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=0.8, shrinkA=2, shrinkB=2,
                                                     connectionstyle="arc3,rad=-0.25", mutation_scale=8), zorder=7)
    ax.plot(*p, marker="o", ms=3.5, mfc="white", mec=S.INK, mew=0.8, zorder=7, clip_on=False)
    ax.plot(*fp, marker="o", ms=3.5, mfc=S.INK, mec=S.INK, zorder=7)
    ax.text(p[0] + 0.8, p[1] + 1.2, "$(10, 0)$", fontsize=8, ha="left", va="bottom")
    ax.text(fp[0], fp[1] + 1.4, "$f(10, 0)$", fontsize=8, ha="center", va="bottom")
    ax.text(c[2] + 0.8, 29.2, r"$\Phi(L)$", fontsize=8, ha="left", va="top")
    ax.text(cc[2] + 0.8, 61.5, r"$\Phi(C)$", fontsize=8, ha="left", va="top")
    ax.text(h[2] - 0.8, h[3] + 0.8, r"hull of $f(L)$", fontsize=8, ha="right", va="bottom")

    handles = [S.patch(r"$\Phi(L)$", S.INK, alpha=0.18),
               Line2D([], [], ls="none", marker="o", ms=3.2, mfc=S.INK, mec="white", label=r"$f$ at the corners of $L$"),
               Line2D([], [], color=S.INK, lw=1.0, ls=(0, (5, 3)), label=r"hull of $f(L)$"),
               Line2D([], [], ls="none", marker=".", ms=4, color=S.LIGHT_GRAY, label=r"$f$ at 8000 points of $L$"),
               Line2D([], [], color=S.INK, lw=1.0, ls=(0, (1, 1.5)), label=r"$\Phi(C)$ = hull of $f(C)$")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.8, borderaxespad=0)
    out = LC.FIGURES / "leslie_depth_one.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    print(out)


if __name__ == "__main__":
    sys.exit(main())
