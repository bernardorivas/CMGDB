"""Isolation of the set around p* with the hull of f, independent of corner sampling.

Let M_12 and M_16 be the Morse sets that contain p* in the corner runs 12/14
and 16/18.  On the 256 x 256 lattice of depth-16 squares, the script builds
the graph G on the squares of |M_12| with an edge c -> c' when the hull of
f(c) (leslie_common.hull_box, widened by w (1 + max |end point|)) meets c'.
Since f(c) lies in the hull, every orbit of f that stays in |M_12| passes
through squares that have a path in G through them in both directions.  The
script computes, for w = 1e-9, 1e-6 and 1e-4:

  1. the squares of |M_12| on a path of G that is infinite in both directions
     (reachable from a cycle and reaching a cycle), compared with the squares
     of M_16, and their distance from the squares outside |M_12|.
  2. for every square c of M_16, the distance from the hull of f(c) to the
     squares outside M_16 (forward invariance of |M_16| with a margin).
and also
  3. the Betti numbers of |M_12| and |M_16|.
  4. the cells of M_12 (on the 12/14 grid) whose hull leaves M_12, and the
     squares of |M_12| whose hull leaves |M_12|.
  5. the distance from Gamma and from p* to the cells outside M_12 and M_16.
  6. that M_16 is also the Morse set around p* of the uniform depth-16 run.

Output: outputs/leslie_isolation_<tag>.txt and .json.  Runs on every build, in
about 2 s on the Apple silicon machine of the stored outputs.  numpy only.

usage: python leslie_isolation.py --tag TAG
"""

import math
import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
from leslie_common import LOWER, UPPER, P_STAR, hull_box


def morse_set_with(CMGDB, model, point):
    mg, gr = CMGDB.ComputeConleyMorseGraph(model)
    N = gr.num_vertices()
    boxes = [list(mg.phase_space_box(i)) for i in range(N)]
    v = next(v for v in range(mg.num_vertices())
             if any(LC.contains_point(boxes[c], point) for c in mg.morse_set(v)))
    return mg, gr, boxes, v, sorted(mg.morse_set(v))


def distance_to_cells(points, boxes):
    """Smallest distance from the points to the closed boxes."""
    P = np.asarray(points, float)
    B = np.asarray(boxes, float)
    best = np.inf
    for k in range(0, len(P), 256):
        p = P[k:k + 256]
        dx = np.maximum(0, np.maximum(B[None, :, 0] - p[:, None, 0], p[:, None, 0] - B[None, :, 2]))
        dy = np.maximum(0, np.maximum(B[None, :, 1] - p[:, None, 1], p[:, None, 1] - B[None, :, 3]))
        best = min(best, float(np.sqrt(dx ** 2 + dy ** 2).min()))
    return best


def main():
    import CMGDB
    args = LC.parse_args(__doc__)
    out = LC.Output("leslie_isolation", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    rec = {"cmgdb_version": LC.cmgdb_version(), "tag": args.tag}

    mg12, gr12, boxes12, v12, M12 = morse_set_with(CMGDB, LC.leslie_model(12, 14), P_STAR)
    mg16, gr16, boxes16, v16, M16 = morse_set_with(CMGDB, LC.leslie_model(16, 18), P_STAR)
    out(f"M_12 = Morse set {v12} of the corner 12/14 run: {len(M12)} cells, annotation {list(mg12.annotations(v12))}")
    out(f"M_16 = Morse set {v16} of the corner 16/18 run: {len(M16)} cells, annotation {list(mg16.annotations(v16))}")

    n = 256
    wx, wy = (UPPER[0] - LOWER[0]) / n, (UPPER[1] - LOWER[1]) / n
    S = LC.lattice_squares([boxes12[c] for c in M12], 16)
    T = LC.lattice_squares([boxes16[c] for c in M16], 16)
    out(f"depth-16 squares: |M_12| has {len(S)}, |M_16| has {len(T)}, |M_16| inside |M_12|: {T <= S}")
    rec.update({"M12_cells": len(M12), "M16_cells": len(M16), "M12_squares": len(S), "M16_squares": len(T),
                "M16_inside_M12": T <= S})

    def square(c):
        i, j = c
        return [LOWER[0] + i * wx, LOWER[1] + j * wy, LOWER[0] + (i + 1) * wx, LOWER[1] + (j + 1) * wy]

    def squares_meeting(h):
        i0 = math.floor((h[0] - LOWER[0]) / wx - 1e-9)
        i1 = math.floor((h[2] - LOWER[0]) / wx + 1e-9)
        j0 = math.floor((h[1] - LOWER[1]) / wy - 1e-9)
        j1 = math.floor((h[3] - LOWER[1]) / wy + 1e-9)
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def distance_to_outside(h, region):
        """Distance from the box h to the squares near h that are not in region
        (0 when h meets one of them)."""
        i0, i1 = math.floor((h[0] - LOWER[0]) / wx), math.floor((h[2] - LOWER[0]) / wx)
        j0, j1 = math.floor((h[1] - LOWER[1]) / wy), math.floor((h[3] - LOWER[1]) / wy)
        best = math.inf
        for i in range(i0 - 2, i1 + 3):
            for j in range(j0 - 2, j1 + 3):
                if (i, j) in region:
                    continue
                b = square((i, j))
                dx = max(0.0, b[0] - h[2], h[0] - b[2])
                dy = max(0.0, b[1] - h[3], h[1] - b[3])
                best = min(best, math.hypot(dx, dy))
        return best

    cells = sorted(S)
    pos = {c: k for k, c in enumerate(cells)}
    rec["widenings"] = {}
    for w in (1e-9, 1e-6, 1e-4):
        adj = []
        for c in cells:
            adj.append(sorted(pos[t] for t in squares_meeting(hull_box(square(c), widen=w)) if t in pos))
        rec_cells = set(x for comp in LC.recurrent_components(adj) for x in comp)
        radj = [[] for _ in cells]
        for k, ts in enumerate(adj):
            for t in ts:
                radj[t].append(k)
        inv = LC.reachable(adj, rec_cells) & LC.reachable(radj, rec_cells)
        inv_sq = {cells[k] for k in inv}

        def cheb(c, R=8):
            for r in range(1, R + 1):
                ring = [(c[0] + di, c[1] + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                        if max(abs(di), abs(dj)) == r]
                if any(t not in S for t in ring):
                    return r
            return R + 1
        dmin = min(cheb(c) for c in inv_sq)
        margins = [distance_to_outside(hull_box(square(c), widen=w), T) for c in sorted(T)]
        out("", f"widening w = {w:g}:")
        out(f"   squares of |M_12| on a path of G infinite in both directions: {len(inv_sq)}; equal to the squares "
            f"of M_16: {inv_sq == T}")
        out(f"   smallest Chebyshev distance (in squares) from them to a square outside |M_12|: {dmin}")
        out(f"   hull of f(c) for c in M_16: meets a square outside M_16 for {sum(m == 0 for m in margins)} squares; "
            f"smallest distance to the squares outside M_16: {min(margins):.3e}")
        rec["widenings"][f"{w:g}"] = {"invariant_squares": len(inv_sq), "equal_M16": inv_sq == T,
                                      "chebyshev_distance_to_outside_M12": dmin,
                                      "M16_squares_leaving": int(sum(m == 0 for m in margins)),
                                      "M16_margin": min(margins)}

    # 3. Betti numbers -------------------------------------------------------------
    b12, b16 = LC.betti_numbers(S), LC.betti_numbers(T)
    out("", f"Betti numbers (b0, b1, b2): |M_12| {b12},  |M_16| {b16}")
    rec["betti"] = {"M12": b12, "M16": b16}

    # 4. Where the hull leaves M_12 ------------------------------------------------------
    Ms = set(M12)
    N12 = gr12.num_vertices()
    cC = next(i for i in range(N12) if LC.depth_of(boxes12[i]) == 1)
    leave, into_C = 0, 0
    for c in M12:
        h = hull_box(boxes12[c])
        hit = [t for t in range(N12) if LC.boxes_meet(h, boxes12[t])]
        if any(t not in Ms for t in hit):
            leave += 1
        if cC in hit:
            into_C += 1
    leave16 = sum(1 for c in cells if any(t not in S for t in squares_meeting(hull_box(square(c)))))
    out(f"cells of M_12 (12/14 grid) whose hull meets a cell outside M_12: {leave} of {len(M12)} ({into_C} of them meet C);"
        f"  squares of |M_12| whose hull meets a square outside |M_12|: {leave16} of {len(S)}")
    rec.update({"M12_cells_leaving": leave, "M12_cells_into_C": into_C, "M12_squares_leaving": leave16})

    # 5. Distances ------------------------------------------------------------------------
    gamma = LC.gamma_orbit(4124, step=97)
    for name, boxes, Mc in (("M_12", boxes12, M12), ("M_16", boxes16, M16)):
        Mset = set(Mc)
        others = [boxes[i] for i in range(len(boxes)) if i not in Mset]
        dg, dp = distance_to_cells(gamma, others), distance_to_cells([P_STAR], others)
        out(f"distance to the cells outside {name}: from Gamma (4124 points) {dg:.4f}, from p* {dp:.4f}")
        rec[f"distances_{name}"] = {"Gamma": dg, "p*": dp}

    # 6. The uniform depth-16 run --------------------------------------------------------------
    mgu, gru, boxesu, vu, Mu = morse_set_with(CMGDB, LC.leslie_model(16, 16, 16), P_STAR)
    same = sorted(tuple(boxesu[c]) for c in Mu) == sorted(tuple(boxes16[c]) for c in M16)
    out(f"uniform depth 16: the Morse set {vu} around p* has {len(Mu)} cells, annotation {list(mgu.annotations(vu))}; "
        f"same boxes as M_16: {same}")
    rec["uniform16_equal_M16"] = same
    out.json(rec)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
