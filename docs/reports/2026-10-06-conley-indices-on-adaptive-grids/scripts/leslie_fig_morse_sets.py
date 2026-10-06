"""Figure: Morse sets at depth 16 for three box maps and two hierarchies.

(a) the run 16/18 with Phi, the box spanned by f at the corners of B,
(b) the uniform run of depth 16 with Phi, (c) the run 16/18 with Phi_hull,
the hull of f(B), and (d) the run 16/18 with Phi_pad, Phi(B) widened on
each side by the widths of B.  Morse
sets are colored by the invariant sets they contain, cells of depth < 16
(cells that the decomposition does not refine) are shaded, and Gamma, p*,
0, a3 and s3 are drawn on top.  Each panel lists the annotations.

Reads outputs/leslie_morse_<tag>.json and outputs/leslie_dynamics.json and
writes figures/leslie_morse_sets_16.pdf.

usage: python leslie_fig_morse_sets.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import leslie_common as LC
import leslie_plotstyle as S
from leslie_plotstyle import plt, Line2D

PANELS = [("corner 16/18", r"(a) $\Phi$, run $16/18$"),
          ("corner uniform 16", r"(b) $\Phi$, uniform run of depth $16$"),
          ("hull 16/18", r"(c) $\Phi_{\mathrm{hull}}$, run $16/18$"),
          ("padded 16/18", r"(d) $\Phi_{\mathrm{pad}}$, run $16/18$")]


def color_of(label):
    names = set(n.strip() for n in label.split(","))
    if names == {"p*", "Gamma"}:
        return S.BLUE
    if names == {"a3"}:
        return S.ORANGE
    if names == {"s3"}:
        return S.AQUA
    if names == {"p*", "s3", "Gamma"}:
        return S.VIOLET
    if names == {"0"}:
        return S.DARK_GRAY
    if names == {"none of these"}:
        return S.LIGHT_GRAY
    raise ValueError(f"no color for a Morse set that contains {label}")


SHORT = {"p*, Gamma": r"$p^{\ast}\!,\Gamma$", "a3": r"$a_3$", "s3": r"$s_3$",
         "p*, s3, Gamma": r"$p^{\ast}\!,\Gamma, s_3$",
         "0": r"$0$"}


def annotation_tex(ann):
    return "$(" + r",\ ".join(a.replace("^", "^") for a in ann) + ")$"


def main():
    args = LC.parse_args(__doc__)
    S.setup()
    data = LC.load_json("leslie_morse", args.tag)
    runs = {r["label"]: r for r in data["runs"]}
    dyn = S.load_dynamics()
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXT_WIDTH, 5.6), gridspec_kw=dict(hspace=0.28, wspace=0.16))
    for ax, (key, title) in zip(axs.ravel(), PANELS):
        r = runs[key]
        S.domain_axes(ax)
        if "boxes" in r:
            boxes = r["boxes"]
            coarse = [b for b in boxes if LC.depth_of(b) < r["min"]]
            S.add_boxes(ax, coarse, facecolor=S.PALE, edgecolor=S.LIGHT_GRAY, lw=0.3, zorder=1)
            get = lambda s: [boxes[c] for c in s["cells"]]
        else:
            get = lambda s: s["boxes"]
        lines = []
        for s in r["morse_sets"]:
            col = color_of(s["label"])
            S.add_boxes(ax, get(s), facecolor=col, edgecolor=col, lw=0.15, zorder=3)
            if s["label"] == "none of these":
                xs = [(b[0] + b[2]) / 2 for b in get(s)]
                ys = [(b[1] + b[3]) / 2 for b in get(s)]
                ax.plot(xs, ys, ls="none", marker="x", ms=3.2, mew=0.6, color=S.MUTED, zorder=4)
            else:
                lines.append(f"{SHORT[s['label']]}: {annotation_tex(s['annotation'])}  [{s['num_cells']}]")
        S.plot_invariant_sets(ax, dyn, zorder=8)
        none = sum(1 for s in r["morse_sets"] if s["label"] == "none of these")
        if none:
            lines.append(f"{none} sets with none: $(0,\\ 0,\\ 0)$")
        ax.text(0.985, 0.975, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7,
                linespacing=1.35, zorder=10, bbox=dict(fc="white", ec=S.LIGHT_GRAY, lw=0.5, pad=2.0, alpha=0.92))
        if key == "corner 16/18":
            ax.text(67.5, 30.0, "$C$", ha="center", va="center", fontsize=10, color=S.MUTED, zorder=5)
            ax.text(22.5, 46.0, "$C$\u2032", ha="center", va="center", fontsize=10, color=S.MUTED, zorder=5)
        S.panel_label(ax, title)
        if key in ("corner 16/18", "corner uniform 16"):
            ax.set_xlabel("")
        if key in ("corner uniform 16", "padded 16/18"):
            ax.set_ylabel("")
    handles = [S.patch(r"contains $p^{\ast}$ and $\Gamma$", S.BLUE), S.patch(r"contains $a_3$", S.ORANGE),
               S.patch(r"contains $s_3$", S.AQUA), S.patch(r"contains $p^{\ast}$, $\Gamma$, $s_3$", S.VIOLET),
               S.patch(r"contains $0$", S.DARK_GRAY),
               Line2D([], [], ls="none", marker="x", ms=4, mew=0.7, color=S.MUTED, label="contains none of these"),
               S.patch("cell of depth < 16", S.PALE, edge=S.LIGHT_GRAY, lw=0.5)] + S.invariant_set_handles()
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.045), handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4)
    out = LC.FIGURES / "leslie_morse_sets_16.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    print(out)


if __name__ == "__main__":
    sys.exit(main())
