"""The fixed point 0 of the Leslie map and the lower-left corner of the domain.

Df(0) has the unstable eigenvalue 20.412 with eigenvector (1, 0.0343) and the
stable eigenvalue -0.812 with eigenvector (1, -0.862).  For t >= 0, the points
-t (1, 0.0343) lie in Q = [-0.001, 90] x [-0.001, 70] only for t <= 0.001, up to
the side x = -0.001.  The script computes the Morse set that contains 0, with
the corner box map at uniform depth 14, on six domains [x_lo, 90] x [y_lo, 70],
and reports its cells, whether it touches the left or lower side of the domain,
its annotation, ComputeConleyIndexForCells, and the components of its exit set
A = cover F(M) minus M, together with how far each of the four half-lines from 0
along the eigenvectors stays in the domain.  For each domain it also reports
the cell of 0, the cells of A whose interiors meet the segment -t (1, 0.0343),
0 < t < t_max, where t_max is the largest t with -t (1, 0.0343) in the domain,
and the point (0.9 x_c, 0), where x_c is the left end of the cell of 0, with its
image under f.  This point lies in the interior of the cell, and when its image
leaves the domain, (|X|, |A|) violates condition (iii) of an index pair,
f(|X| minus |A|) inside |X|.  It also reports the pair of the origin cell in the
corner 12/14 run on Q, with the ranks of H_k(X, A) over F_5.

Output: outputs/leslie_origin_<tag>.txt and .json.  Runs on every build, in
about 1 s on the Apple silicon machine of the stored outputs.  numpy only.

usage: python leslie_origin.py --tag TAG
"""

import math
import sys

sys.dont_write_bytecode = True

import leslie_common as LC
from leslie_common import THETA, fmt_box, histogram

DOMAINS = [((-0.001, -0.001), (90.0, 70.0)), ((-0.001, -1.0), (90.0, 70.0)), ((-1.0, -0.001), (90.0, 70.0)),
           ((-1.0, -0.05), (90.0, 70.0)), ((-1.0, -1.0), (90.0, 70.0)), ((-5.0, -5.0), (90.0, 70.0))]


def components(boxes, cells):
    cells = sorted(cells)
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            if LC.boxes_meet(boxes[a], boxes[b], tol=1e-9):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comps = {}
    for c in cells:
        comps.setdefault(find(c), []).append(c)
    return sorted(comps.values(), key=lambda c: (-len(c), c))


def stays_inside(v, lo):
    """Largest t >= 0 with t v in [lo_x, inf) x [lo_y, inf)."""
    ts = [lo[k] / v[k] for k in (0, 1) if v[k] < 0]
    return min(ts) if ts else math.inf


def segment_meets_interior(v, tmax, box):
    """Whether t v, 0 < t < tmax, meets the interior of box [x0, y0, x1, y1]."""
    lo, hi = 0.0, tmax
    for k in (0, 1):
        a, b = box[k], box[k + 2]
        if v[k] == 0.0:
            if not a < 0.0 < b:
                return False
            continue
        ta, tb = a / v[k], b / v[k]
        lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
    return lo < hi


def inside(p, lo, hi):
    return lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]


def origin_pair(CMGDB, model):
    mg, gr = CMGDB.ComputeConleyMorseGraph(model)
    N = gr.num_vertices()
    boxes = [list(mg.phase_space_box(i)) for i in range(N)]
    v = next(v for v in range(mg.num_vertices()) if any(LC.contains_point(boxes[c], (0.0, 0.0)) for c in mg.morse_set(v)))
    M = sorted(mg.morse_set(v))
    X = set()
    for c in M:
        X.update(gr.adjacencies(c))
    A = sorted(X - set(M))
    comps = components(boxes, A)
    try:
        idx = list(CMGDB.ComputeConleyIndexForCells(model, mg, M))
    except Exception as e:
        idx = f"{type(e).__name__}: {e}"[:200]
    return mg, boxes, v, M, X, A, comps, idx


def c0_interior(box):
    """Whether 0 lies in the interior of box."""
    return box[0] < 0.0 < box[2] and box[1] < 0.0 < box[3]


def main():
    import CMGDB
    args = LC.parse_args(__doc__)
    out = LC.Output("leslie_origin", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    a, b = THETA
    lu = (a + math.sqrt(a * a + 2.8 * b)) / 2
    ls = (a - math.sqrt(a * a + 2.8 * b)) / 2
    vu, vs = (1.0, 0.7 / lu), (1.0, 0.7 / ls)
    out(f"Df(0): unstable eigenvalue {lu:.6f}, eigenvector (1, {vu[1]:.6f}); stable eigenvalue {ls:.6f}, "
        f"eigenvector (1, {vs[1]:.6f})")
    rec = {"cmgdb_version": LC.cmgdb_version(), "tag": args.tag, "domains": []}

    for lo, hi in DOMAINS:
        model = LC.leslie_model(14, 14, 14, "corner", lo, hi)
        mg, boxes, v, M, X, A, comps, idx = origin_pair(CMGDB, model)
        BM = LC.bounding_box([boxes[c] for c in M])
        touch = any(abs(boxes[c][0] - lo[0]) < 1e-12 or abs(boxes[c][1] - lo[1]) < 1e-12 for c in M)
        reach = {"+E^u": stays_inside(vu, lo), "-E^u": stays_inside((-vu[0], -vu[1]), lo),
                 "+E^s": stays_inside(vs, lo), "-E^s": stays_inside((-vs[0], -vs[1]), lo)}
        out("", f"domain [{lo[0]}, {hi[0]}] x [{lo[1]}, {hi[1]}], uniform depth 14:")
        out("   t v in [x_lo, inf) x [y_lo, inf) for t up to: "
            + ", ".join(f"{k} {'all t >= 0' if t == math.inf else f'{t:.4g}'}" for k, t in reach.items()))
        out(f"   Morse set {v} contains 0: {len(M)} cells in {fmt_box(BM, 3)}, touches the left or lower side: {touch}")
        out(f"   annotation {list(mg.annotations(v))};  ComputeConleyIndexForCells {idx}")
        desc = [f"{len(K)} cells in {fmt_box(LC.bounding_box([boxes[c] for c in K]), 3)}" for K in comps]
        out(f"   exit set A = cover F(M) minus M: {len(A)} cells in {len(comps)} component(s): {desc}")
        c0 = next(boxes[c] for c in M if c0_interior(boxes[c]))
        vneg = (-vu[0], -vu[1])
        seg = [boxes[c] for c in A if segment_meets_interior(vneg, reach["-E^u"], boxes[c])]
        out(f"   cell of 0: {fmt_box(c0, 3)}")
        out(f"   cells of A whose interiors meet -t v_u, 0 < t < {reach['-E^u']:.4g}: {len(seg)}"
            + (": " + ", ".join(fmt_box(b, 3) for b in seg) if seg else ""))
        p = (0.9 * c0[0], 0.0)
        fp = LC.f(p)
        p_inner = c0[0] < p[0] < c0[2] and c0[1] < p[1] < c0[3]
        out(f"   point (0.9 x_c, 0) = ({p[0]:.6g}, 0) in the interior of the cell of 0: {p_inner}; "
            f"f = ({fp[0]:.6f}, {fp[1]:.6f}), outside the domain: {not inside(fp, lo, hi)}")
        rec["domains"].append({"lower": list(lo), "upper": list(hi), "half_lines": {k: (None if t == math.inf else t)
                                                                                  for k, t in reach.items()},
                               "cells": len(M), "bbox": BM, "touches_left_or_lower_side": touch,
                               "annotation": list(mg.annotations(v)), "index_for_cells": idx, "A": len(A),
                               "A_components": [len(K) for K in comps], "cell_of_0": c0,
                               "A_cells_on_unstable_segment": seg,
                               "point_of_cell_of_0": {"point": list(p), "interior": p_inner, "image": list(fp),
                                                      "image_outside_domain": not inside(fp, lo, hi)}})

    model = LC.leslie_model(12, 14)
    mg, boxes, v, M, X, A, comps, idx = origin_pair(CMGDB, model)
    sq = lambda cells: LC.lattice_squares([boxes[c] for c in cells], 12)
    h = LC.relative_homology_ranks(sq(X), sq(A))
    depths = [LC.depth_of(boxes[c]) for c in A]
    out("", f"corner 12/14 on Q: Morse set {v} = {M} contains 0; |X| = {len(X)}, |A| = {len(A)} (depths "
        f"{histogram(depths)}) in {len(comps)} component(s); H_k(X, A; F_5) has ranks {h}; annotation "
        f"{list(mg.annotations(v))}")
    rec["corner_12_14"] = {"M": M, "X": len(X), "A": len(A), "A_components": [len(K) for K in comps],
                           "relative_homology": h, "annotation": list(mg.annotations(v))}
    out.json(rec)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
