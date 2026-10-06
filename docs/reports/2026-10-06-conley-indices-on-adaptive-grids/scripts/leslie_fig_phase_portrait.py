"""Figure: phase portrait of the Leslie map on Q and the basins of a3 and Gamma.

(a) the fixed points 0 and p*, the invariant circle Gamma, the 3-cycles a3
and s3, and the two branches of W^u(s3) at each point of s3 (one goes to a3,
the other to Gamma).  The dashed line x = 44.9995 separates the cells L and C
of depth 1.  (b) the seeds of a 361 x 281 grid that go to a3 and to Gamma.

Reads outputs/leslie_dynamics.json and writes figures/leslie_phase_portrait.pdf.

usage: python leslie_fig_phase_portrait.py
"""

import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
import leslie_plotstyle as S
from leslie_plotstyle import plt, Line2D


def main():
    LC.parse_args(__doc__)
    S.setup()
    dyn = S.load_dynamics()
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_WIDTH, 3.05), gridspec_kw=dict(wspace=0.18))
    xm = (LC.LOWER[0] + LC.UPPER[0]) / 2

    ax = axs[0]
    S.domain_axes(ax)
    ax.axvline(xm, color=S.LIGHT_GRAY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(xm - 1.5, 67.5, "$L$", ha="right", va="top", color=S.MUTED)
    ax.text(xm + 1.5, 67.5, "$C$", ha="left", va="top", color=S.MUTED)
    for br in dyn["unstable_branches_s3"]:
        P = np.array(br["curve"])
        ax.plot(P[:, 0], P[:, 1], lw=0.6, color=S.ORANGE if br["fate"] == "a3" else S.BLUE, zorder=3)
    S.plot_invariant_sets(ax, dyn)
    S.panel_label(ax, r"(a) invariant sets and $W^u(s_3)$")

    ax = axs[1]
    S.domain_axes(ax)
    b = dyn["basins"]
    lab = np.array([[int(ch) for ch in row] for row in b["rows"]])
    xs = np.linspace(b["xs"][0], b["xs"][1], b["xs"][2])
    ys = np.linspace(b["ys"][0], b["ys"][1], b["ys"][2])
    ax.contourf(xs, ys, (lab == 1).astype(float), levels=[0.5, 1.5], colors=[S.ORANGE], alpha=0.32, zorder=1,
                antialiased=True)
    ax.contourf(xs, ys, (lab == 2).astype(float), levels=[0.5, 1.5], colors=[S.BLUE], alpha=0.32, zorder=1,
                antialiased=True)
    S.plot_invariant_sets(ax, dyn)
    S.panel_label(ax, r"(b) basins of $a_3$ and $\Gamma$")

    handles = S.invariant_set_handles() + [
        Line2D([], [], color=S.ORANGE, lw=1.0, label=r"$W^u(s_3)\to a_3$"),
        Line2D([], [], color=S.BLUE, lw=1.0, label=r"$W^u(s_3)\to\Gamma$"),
        S.patch(r"basin of $a_3$", S.ORANGE, alpha=0.32),
        S.patch(r"basin of $\Gamma$", S.BLUE, alpha=0.32)]
    fig.legend(handles=handles, loc="lower center", ncol=9, bbox_to_anchor=(0.5, -0.035), handlelength=1.4,
               columnspacing=0.9, handletextpad=0.4)
    out = LC.FIGURES / "leslie_phase_portrait.pdf"
    LC.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    print(out)


if __name__ == "__main__":
    sys.exit(main())
