"""The nodes of depths 1 and 2 of the hierarchy for the Leslie map, and the cell C.

Every run with init 0 starts from the same decompositions at depths 1 and 2,
so Model(k, k, lower, upper, BoxMap(f, B)) shows them.  The script prints,
with the corner box map:

  1. the cells of Model(1, 1) and Model(2, 2) with their corner images, their
     padded images and the hulls of their images under f, the edges of the
     final graph and the Morse sets.
  2. the corner image of L = [-0.001, 44.9995] x [-0.001, 70] against the hull
     of f(L), with the points where f1 is largest on L and f(10, 0), which
     lies in C = [44.9995, 90] x [-0.001, 70].
  3. the check that f1 decreases in x and in y on C, so that the corner image
     of C is the hull of f(C).
  4. the chain of boxes of depth 0, ..., 12 that contain p = (7.7, 16.9), a
     point of a cell of the 12/14 Morse set that maps into C, with the corner,
     padded and hull x-ranges of their images.

Output: outputs/leslie_depth_one_<tag>.txt and .json.  Runs on every build
(only Model, ComputeMorseGraph and BoxMap are used), in under 1 s.

usage: python leslie_depth_one.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import leslie_common as LC
from leslie_common import LOWER, UPPER, THETA, f, fmt_box, hull_box, corner_box_map, padded_box_map


def main():
    import CMGDB
    args = LC.parse_args(__doc__)
    out = LC.Output("leslie_depth_one", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    xm = (LOWER[0] + UPPER[0]) / 2
    ym = (LOWER[1] + UPPER[1]) / 2
    L = [LOWER[0], LOWER[1], xm, UPPER[1]]
    C = [xm, LOWER[1], UPPER[0], UPPER[1]]
    Cp = [LOWER[0], ym, xm, UPPER[1]]
    Lb = [LOWER[0], LOWER[1], xm, ym]
    rec = {"L": L, "C": C, "C'": Cp, "L_b": Lb}
    out(f"L = {fmt_box(L)},  C = {fmt_box(C)},  C' = {fmt_box(Cp)},  L_b = {fmt_box(Lb)}")

    # 1. Model(1, 1) and Model(2, 2) --------------------------------------------
    rec["depths"] = {}
    for k in (1, 2):
        mg, gr = CMGDB.ComputeMorseGraph(LC.leslie_model(k, k))
        N = gr.num_vertices()
        cells = []
        out("", f"1. Model({k}, {k}, lower, upper, corner box map): {N} cells")
        for i in range(N):
            b = list(mg.phase_space_box(i))
            row = {"box": b, "depth": LC.depth_of(b), "corner": corner_box_map(b), "padded": padded_box_map(b),
                   "hull": hull_box(b, widen=0.0), "edges": sorted(gr.adjacencies(i))}
            cells.append(row)
            out(f"   cell {i}: depth {row['depth']}, {fmt_box(b)}")
            out(f"      corner image {fmt_box(row['corner'])};  hull of f(cell) {fmt_box(row['hull'])}")
            out(f"      padded image {fmt_box(row['padded'])};  edges of the final graph -> {row['edges']}")
        ms = [sorted(mg.morse_set(v)) for v in range(mg.num_vertices())]
        out(f"   Morse sets {ms}")
        rec["depths"][str(k)] = {"cells": cells, "morse_sets": ms}

    # 2. The image of L ----------------------------------------------------------
    a, b = THETA
    yc = 10.0 - a / b * L[0]                     # maximizer of f1 on the edge x = x0 of L
    vmax = f((L[0], yc))[0]
    corner_L, hull_L, pad_L = corner_box_map(L), hull_box(L, widen=0.0), padded_box_map(L)
    corners = [(x, y) for x in (L[0], L[2]) for y in (L[1], L[3])]
    out("", "2. The image of L.")
    for p in corners:
        q = f(p)
        out(f"   f({p[0]}, {p[1]}) = ({q[0]:.4f}, {q[1]:.4f})")
    out(f"   corner image Phi(L) = {fmt_box(corner_L)}  (meets C: {corner_L[2] >= C[0]})")
    out(f"   padded image        = {fmt_box(pad_L)}  (meets C: {pad_L[2] >= C[0]})")
    out(f"   hull of f(L)        = {fmt_box(hull_L)}  (meets C: {hull_L[2] >= C[0]})")
    out(f"   max f1 on L = f1({L[0]}, {yc:.6f}) = {vmax:.4f};  f(0, 10) = ({f((0.0, 10.0))[0]:.4f}, 0);  "
        f"f(10, 0) = ({f((10.0, 0.0))[0]:.4f}, {f((10.0, 0.0))[1]:.4f}), in C: {LC.contains_point(C, f((10.0, 0.0)))}")
    rec["image_of_L"] = {"corners": [list(p) for p in corners], "corner_images": [f(p) for p in corners],
                         "corner": corner_L, "padded": pad_L, "hull": hull_L,
                         "argmax": [L[0], yc], "max": vmax, "f(10,0)": f((10.0, 0.0))}

    # 3. f1 on C -----------------------------------------------------------------
    smin = a * C[0] + b * C[1]
    out("", "3. On C, s = a x + b y >= " f"{smin:.3f} > 10 b = {10 * b:.1f} > 10 a = {10 * a:.1f}, so "
        "d f1/dx = (a - 0.1 s) e < 0 and d f1/dy = (b - 0.1 s) e < 0:")
    out(f"   f1 decreases in x and in y on C, and the corner image of C, {fmt_box(corner_box_map(C))}, "
        f"is the hull of f(C), {fmt_box(hull_box(C, widen=0.0))}")
    rec["image_of_C"] = {"corner": corner_box_map(C), "hull": hull_box(C, widen=0.0), "s_min": smin}

    # 4. Ancestor chain of the cell containing p ---------------------------------
    p = (7.7, 16.9)
    chain = LC.ancestor_chain(p, 12)
    out("", f"4. Boxes of depth 0, ..., 12 that contain p = {p}; x-ranges of their images "
        "(corner: Phi(B), padded, and the hull of f(B)):")
    rows = []
    prev = None
    for d, B in enumerate(chain):
        c, pd, h = corner_box_map(B), padded_box_map(B), hull_box(B, widen=0.0)
        outer = LC.box_inside(h, c)
        mono = prev is None or LC.box_inside(c, prev)
        rows.append({"depth": d, "box": B, "corner": c, "padded": pd, "hull": h, "corner_is_outer": outer,
                     "corner_inside_parent_corner": mono, "corner_meets_C": c[2] >= C[0]})
        out(f"   depth {d:2d}  {fmt_box(B)}  corner [{c[0]:8.4f}, {c[2]:8.4f}]  padded [{pd[0]:9.4f}, {pd[2]:8.4f}]  "
            f"hull [{h[0]:8.4f}, {h[2]:8.4f}]  " + ("" if outer else "corner misses f(B)  ")
            + ("" if mono else "corner image not inside the parent's  ") + ("meets C" if c[2] >= C[0] else ""))
        prev = c
    rec["chain"] = {"point": list(p), "rows": rows}
    out.json({"cmgdb_version": LC.cmgdb_version(), "tag": args.tag, **rec})
    out.close()


if __name__ == "__main__":
    sys.exit(main())
