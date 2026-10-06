"""Figure: the final grid of the run 12/14 with Phi and the cycle M -> C -> M.

Draws the 636 cells of Model(12, 14, lower, upper, BoxMap(f, B)), the Morse
set M around p* (525 cells) with the 28 cells that map into C and the 6
cells that C maps into, the cell that contains 0, the cells of
A = cover Phi(M) minus M (eight small cells and C), the cell C' of depth 2,
the boxes Phi(c_p), for the cell c_p of M that contains p, and Phi(C)
(dashed), and the points p, f(p), q, f(q) that realize the edges M -> C and
C -> M.

Reads outputs/leslie_cycle_<tag>.json and outputs/leslie_dynamics.json and writes
figures/leslie_cycle_12_14.pdf.

usage: python leslie_fig_cycle.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import leslie_common as LC
import leslie_plotstyle as S
from leslie_plotstyle import plt, Line2D


def main():
    args = LC.parse_args(__doc__)
    S.setup()
    d = LC.load_json("leslie_cycle", args.tag)
    dyn = S.load_dynamics()
    B = d["boxes"]
    M, into_C, from_C = d["M"], set(d["M_cells_into_C"]), set(d["C_into_M"])
    cC, cCp = d["C"], d["C'"]
    K1 = [a for a in d["A"] if a != cC]

    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH, 4.75))
    S.domain_axes(ax)
    S.add_boxes(ax, [B[m] for m in M if m not in into_C], facecolor=S.BLUE, alpha=0.22, edgecolor="none", zorder=1)
    S.add_boxes(ax, [B[m] for m in into_C], facecolor=S.BLUE, alpha=0.75, edgecolor="none", zorder=1)
    S.add_boxes(ax, [B[c] for c in d["origin_cells"]], facecolor=S.DARK_GRAY, edgecolor="none", zorder=1)
    S.add_boxes(ax, [B[a] for a in K1] + [B[cC]], facecolor="none", edgecolor=S.HATCH, hatch="////", lw=0.0,
                zorder=1)
    S.add_boxes(ax, B, facecolor="none", edgecolor=S.LIGHT_GRAY, lw=0.25, zorder=2)
    S.add_boxes(ax, [B[m] for m in from_C], facecolor="none", edgecolor=S.INK, lw=0.9, zorder=5)
    for b, label in ((d["corner_image_cell_p"], r"$\Phi(c_p)$"), (d["corner_image_C"], r"$\Phi(C)$")):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=S.INK, lw=0.9,
                                   ls=(0, (4, 2)), zorder=6))
    pc = d["corner_image_cell_p"]
    ax.text(pc[2] + 0.6, (pc[1] + pc[3]) / 2, r"$\Phi(c_p)$", fontsize=8, ha="left", va="center", zorder=8,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    cc = d["corner_image_C"]
    ax.text(cc[2] + 0.6, cc[3] - 0.5, r"$\Phi(C)$", fontsize=8, ha="left", va="top", zorder=8,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    ax.text(67.5, 66.0, "$C$", fontsize=11, ha="center", va="top",
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9), zorder=8)
    ax.text(30.0, 66.0, "$C$\u2032", fontsize=11, ha="center", va="top", zorder=8)

    S.plot_invariant_sets(ax, dyn, zorder=9)
    p, fp, q, fq = d["p"], d["f(p)"], d["q"], d["f(q)"]
    for a, b, rad in ((p, fp, -0.25), (q, fq, 0.3)):
        ax.annotate("", xy=b, xytext=a, zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=0.9, shrinkA=2.5, shrinkB=2.5,
                                    connectionstyle=f"arc3,rad={rad}", mutation_scale=8))
    for a, b in ((p, fp), (q, fq)):
        ax.plot(*a, marker="o", ms=3.6, mfc="white", mec=S.INK, mew=0.9, zorder=13)
        ax.plot(*b, marker="o", ms=3.6, mfc=S.INK, mec="white", mew=0.5, zorder=13)
    ax.text(p[0] - 0.8, p[1] + 0.6, "$p$", fontsize=8, ha="right", va="bottom", zorder=13,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.text(fp[0] + 1.2, pc[3] + 0.5, "$f(p)$", fontsize=8, ha="left", va="bottom", zorder=13,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.text(q[0] + 0.9, q[1] + 0.2, "$q$", fontsize=8, ha="left", va="bottom", zorder=13,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.text(fq[0] + 0.9, fq[1] + 0.7, "$f(q)$", fontsize=8, ha="left", va="bottom", zorder=13,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))

    handles = [S.patch(r"$M$ (525 cells)", S.BLUE, alpha=0.22),
               S.patch(r"cells of $M$ that map into $C$", S.BLUE, alpha=0.75),
               S.patch(r"cells of $M$ that $C$ maps into", "none", edge=S.INK, lw=0.9),
               S.patch(r"$A=\mathrm{cover}\,\Phi(M)\setminus M$", "white", edge=S.HATCH, hatch="////", lw=0.0),
               S.patch(r"the cell that contains $0$", S.DARK_GRAY),
               Line2D([], [], color=S.INK, lw=0.9, ls=(0, (4, 2)), label=r"$\Phi(c_p)$, $\Phi(C)$"),
               Line2D([], [], color=S.INK, lw=0.9, marker="o", ms=3.6, mfc="white", mec=S.INK,
                      label=r"$p \mapsto f(p)$, $q \mapsto f(q)$")] + S.invariant_set_handles()
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.6, borderaxespad=0,
              labelspacing=0.55)
    out = LC.FIGURES / "leslie_cycle_12_14.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    print(out)


if __name__ == "__main__":
    sys.exit(main())
