"""The cycle M -> C -> M of the corner 12/14 run and the pair of M.

On the final grid of Model(12, 14, lower, upper, BoxMap(f, B)), let M be the
Morse set that contains p* and C = [44.9995, 90] x [-0.001, 70] the cell of
depth 1.  The script computes

  1. the pair X = cover F(M), A = X minus M: the cells of A, the components of
     |A| and their images, and the cells of A whose image meets M = X minus A
     (the condition F(A) cap X subset A fails exactly there).
  2. the edges M -> C (cells of M whose corner image meets C) and C -> M, and
     two points p, q with p in M, f(p) in C, q in C, f(q) in M, so that f
     itself has both edges, and q in |A| with f(q) in |X| minus |A|.
  3. the depth-12 squares of C whose corner image meets M.
  4. the ranks of H_k(X, A) and H_k(X cup F(A), A cup F(A)) over F_5 (own
     cubical chain complexes), together with H_k(X), H_k(A), ... .
  5. the annotation of M and ComputeConleyIndexForCells(M): master refuses the
     set (ValueError), the earlier builds return a value.
  6. the strongly connected component of the final cell graph that contains M,
     and its pair and index.
  7. for Model(16, 18, ...): the pair of the Morse set that contains p*, the
     pair of the origin cell, and the component of the final graph that
     contains the origin cell.

Output: outputs/leslie_cycle_<tag>.txt and .json.  Runs on every build, in
about 1 s on the Apple silicon machine of the stored outputs.  numpy only.

usage: python leslie_cycle.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import leslie_common as LC
from leslie_common import P_STAR, ORIGIN, f, fmt_box, histogram, depth_of


def index_for_cells(CMGDB, model, mg, cells):
    try:
        return list(CMGDB.ComputeConleyIndexForCells(model, mg, cells))
    except ValueError as e:
        return "ValueError: " + str(e)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def components_of_union(boxes, cells):
    """Connected components of the union of closed cells (cells that touch,
    also at a corner, are in one component)."""
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


def cell_of(boxes, cells, p):
    return [c for c in cells if LC.contains_point(boxes[c], p)]


def short(x):
    return x if isinstance(x, list) else x[:200] + ("..." if len(x) > 200 else "")


def main():
    import CMGDB
    args = LC.parse_args(__doc__)
    out = LC.Output("leslie_cycle", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    rec = {"cmgdb_version": LC.cmgdb_version(), "tag": args.tag}

    model = LC.leslie_model(12, 14)
    mg, gr = CMGDB.ComputeConleyMorseGraph(model)
    N = gr.num_vertices()
    boxes = [list(mg.phase_space_box(i)) for i in range(N)]
    depths = [depth_of(b) for b in boxes]
    adj = [sorted(gr.adjacencies(i)) for i in range(N)]
    nv = mg.num_vertices()
    vM = next(v for v in range(nv) if cell_of(boxes, mg.morse_set(v), P_STAR))
    vO = next(v for v in range(nv) if cell_of(boxes, mg.morse_set(v), ORIGIN))
    M = sorted(mg.morse_set(vM))
    Ms = set(M)
    cC = next(i for i in range(N) if depths[i] == 1)
    cCp = next(i for i in range(N) if depths[i] == 2)
    out(f"Model(12, 14, lower, upper, corner box map): {N} cells; M = Morse set {vM} (contains p*), "
        f"{len(M)} cells; the origin is in Morse set {vO}")
    out(f"C = cell {cC}, depth 1, {fmt_box(boxes[cC])};  C' = cell {cCp}, depth 2, {fmt_box(boxes[cCp])}")
    rec.update({"num_cells": N, "boxes": boxes, "M_vertex": vM, "M": M, "origin_vertex": vO,
                "origin_cells": sorted(mg.morse_set(vO)), "C": cC, "C'": cCp})
    owner = {c: v for v in range(nv) for c in mg.morse_set(v)}
    where = {}
    for name in ("a3", "s3"):
        for k, pt in enumerate(LC.NAMED_POINTS[name]):
            cells = cell_of(boxes, range(N), pt)
            where[f"{name}[{k}]"] = [(c, depths[c], owner.get(c)) for c in cells]
    out("cells that contain the points of a3 and s3 (cell, depth, Morse set or None): "
        + "; ".join(f"{k}: {v}" for k, v in where.items()))
    rec["three_cycle_cells"] = where

    # 1. The pair --------------------------------------------------------------
    X = set()
    for m in M:
        X.update(adj[m])
    A = sorted(X - Ms)
    out("", f"1. X = cover F(M): {len(X)} cells, M subset of X: {Ms <= X};  A = X minus M: {len(A)} cells, "
        f"depths {histogram(depths[a] for a in A)}")
    comps = components_of_union(boxes, A)
    rec_comps = []
    for K in comps:
        img = set()
        for a in K:
            img.update(adj[a])
        rec_comps.append({"cells": K, "image_in_M": sorted(img & Ms), "image_in_A": sorted(img & set(A)),
                          "image_outside_X": sorted(img - X)})
        out(f"   component of |A|: cells {K} (depths {[depths[a] for a in K]}), bounding box "
            f"{fmt_box(LC.bounding_box([boxes[a] for a in K]), 2)}")
        out(f"      F(component) meets M in {len(img & Ms)} cells, A in {sorted(img & set(A))}, "
            f"and {len(img - X)} cells outside X")
    bad = sorted(a for a in A if Ms & set(adj[a]))
    out(f"   cells a of A with F(a) cap M nonempty: {bad};  F(X minus A) subset X: "
        f"{all(set(adj[m]) <= X for m in M)}")
    rec.update({"X": sorted(X), "A": A, "A_components": rec_comps, "A_cells_mapping_into_M": bad})

    # 2. The cycle M -> C -> M -------------------------------------------------
    into_C = sorted(m for m in M if cC in adj[m])
    from_C = sorted(Ms & set(adj[cC]))
    out("", f"2. {len(into_C)} cells of M map into C;  C maps into {len(from_C)} cells of M: {from_C}, "
        f"together {fmt_box(LC.bounding_box([boxes[c] for c in from_C]))}")
    out(f"   F(C) also meets {len(set(adj[cC]) - X)} cells outside X: {sorted(set(adj[cC]) - X)} (C' = {cCp})")
    p, q = (7.7, 16.9), (45.2, 0.5)
    fp, fq = f(p), f(q)
    cp = cell_of(boxes, M, p)
    cq = cell_of(boxes, from_C, fq)
    out(f"   p = {p} lies in cell {cp} of M, {fmt_box(boxes[cp[0]])}, corner image "
        f"{fmt_box(LC.corner_box_map(boxes[cp[0]]))};  f(p) = ({fp[0]:.4f}, {fp[1]:.4f}) lies in C: "
        f"{LC.contains_point(boxes[cC], fp)}")
    out(f"   q = {q} lies in C: {LC.contains_point(boxes[cC], q)};  corner image of C "
        f"{fmt_box(LC.corner_box_map(boxes[cC]))};  f(q) = ({fq[0]:.4f}, {fq[1]:.4f}) lies in cell {cq} of M, "
        f"{fmt_box(boxes[cq[0]])}")
    in_A = [a for a in A if LC.contains_point(boxes[a], fq)]
    out(f"   q in |A| and f(q) in |X| minus |A| (f(q) lies in no cell of A: {not in_A}), so "
        "f(|A|) cap |X| is not contained in |A|")
    rec.update({"M_cells_into_C": into_C, "C_into_M": from_C, "C_outside_X": sorted(set(adj[cC]) - X),
                "p": list(p), "f(p)": fp, "cell_p": cp[0], "q": list(q), "f(q)": fq, "cell_fq": cq[0],
                "corner_image_cell_p": LC.corner_box_map(boxes[cp[0]]), "corner_image_C": LC.corner_box_map(boxes[cC])})

    # 3. Squares of C at depth 12 -----------------------------------------------
    n = 64
    wx, wy = (LC.UPPER[0] - LC.LOWER[0]) / n, (LC.UPPER[1] - LC.LOWER[1]) / n
    hit = []
    for i in range(32, 64):
        for j in range(64):
            sq = [LC.LOWER[0] + i * wx, LC.LOWER[1] + j * wy, LC.LOWER[0] + (i + 1) * wx, LC.LOWER[1] + (j + 1) * wy]
            img = LC.corner_box_map(sq)
            if any(LC.boxes_meet(img, boxes[m]) for m in M):
                hit.append(sq)
    out("", f"3. depth-12 squares of C whose corner image meets |M|: {len(hit)} of 2048" +
        (f", in {fmt_box(LC.bounding_box(hit), 3)}" if hit else ""))
    rec["C_squares_into_M"] = hit

    # 4. Homology ---------------------------------------------------------------
    FA = set()
    for a in A:
        FA.update(adj[a])
    sq = lambda cells: LC.lattice_squares([boxes[c] for c in cells], 12)
    hX, hA = LC.betti_numbers(sq(X)), LC.betti_numbers(sq(A))
    hXA = LC.relative_homology_ranks(sq(X), sq(A))
    hE, hAE = LC.betti_numbers(sq(X | FA)), LC.betti_numbers(sq(set(A) | FA))
    hXAE = LC.relative_homology_ranks(sq(X | FA), sq(set(A) | FA))
    out("", "4. Ranks over F_5 (degrees 0, 1, 2):")
    out(f"   H(X) = {hX},  H(A) = {hA},  H(X, A) = {hXA}")
    out(f"   F(A): {len(FA)} cells;  H(X cup F(A)) = {hE},  H(A cup F(A)) = {hAE},  "
        f"H(X cup F(A), A cup F(A)) = {hXAE}")
    out(f"   |X cup F(A)| = {len(X | FA)} cells, |A cup F(A)| = {len(set(A) | FA)} cells; (X cup F(A)) minus "
        f"(A cup F(A)) = M minus F(A) has {len((X | FA) - (set(A) | FA))} cells; F(A) cap M = {sorted(FA & Ms)}")
    out(f"   so the inclusion (X, A) -> (X cup F(A), A cup F(A)) does not induce an isomorphism in degree 1 "
        f"({hXA[1]} versus {hXAE[1]})")
    rec["homology"] = {"X": hX, "A": hA, "X,A": hXA, "F(A)": sorted(FA), "X+F(A)": hE, "A+F(A)": hAE,
                       "X+F(A),A+F(A)": hXAE, "F(A)_cap_M": sorted(FA & Ms),
                       "X+F(A)_cells": len(X | FA), "A+F(A)_cells": len(set(A) | FA)}

    # 5. Annotation and ComputeConleyIndexForCells --------------------------------
    ann = list(mg.annotations(vM))
    idx = index_for_cells(CMGDB, model, mg, M)
    out("", f"5. annotation of M: {ann};  ComputeConleyIndexForCells(M): {short(idx)}")
    rec.update({"annotation_M": ann, "index_for_cells_M": idx})

    # 6. The component of the final graph that contains M -------------------------
    comps_G = LC.recurrent_components(adj)
    S = next(c for c in comps_G if Ms <= set(c))
    Ss = set(S)
    XS = set()
    for s in S:
        XS.update(adj[s])
    AS = XS - Ss
    badS = sorted(a for a in AS if Ss & set(adj[a]))
    named = {k: [cell_of(boxes, S, p) != [] for p in pts] for k, pts in LC.NAMED_POINTS.items()}
    idxS = index_for_cells(CMGDB, model, mg, S)
    bS = LC.betti_numbers(sq(S))
    out("", f"6. the final cell graph has {len(comps_G)} strongly connected component(s) with a cycle; the one that "
        f"contains M has {len(S)} cells, depths {histogram(depths[c] for c in S)}")
    out(f"   it contains C: {cC in Ss}, C': {cCp in Ss}, the Morse sets {[v for v in range(nv) if set(mg.morse_set(v)) <= Ss]}, "
        f"and the points {named}")
    out(f"   its pair: |X| = {len(XS)}, |A| = {len(AS)}, cells of A whose image meets it: {badS if badS else 'none'};  "
        f"Betti numbers of its union {bS};  ComputeConleyIndexForCells: {short(idxS)}")

    def misses(b):
        h, c = LC.hull_box(b, widen=0.0), LC.corner_box_map(b)
        e = max(c[0] - h[0], c[1] - h[1], h[2] - c[2], h[3] - c[3], 0.0)
        return e / (1.0 + max(abs(v) for v in h + c))
    not_outer = [i for i in range(N) if misses(boxes[i]) > 1e-8]
    out(f"   cells of the final grid whose corner image misses part of f(cell) (by more than 1e-8, relative): "
        f"{[(i, depths[i]) for i in not_outer]} (cell, depth); in this component: {[i in Ss for i in not_outer]}")
    rec["component_of_M"] = {"cells": S, "depths": histogram(depths[c] for c in S), "X": len(XS), "A": len(AS),
                             "A_cells_mapping_in": badS, "betti": bS, "index_for_cells": idxS, "contains": named,
                             "final_cells_not_outer": not_outer}

    # 7. The 16/18 run ------------------------------------------------------------
    model16 = LC.leslie_model(16, 18)
    mg16, gr16 = CMGDB.ComputeConleyMorseGraph(model16)
    N16 = gr16.num_vertices()
    b16 = [list(mg16.phase_space_box(i)) for i in range(N16)]
    d16 = [depth_of(b) for b in b16]
    adj16 = [sorted(gr16.adjacencies(i)) for i in range(N16)]
    out("", f"7. Model(16, 18, lower, upper, corner box map): {N16} cells")
    r16 = {}
    for v in range(mg16.num_vertices()):
        Mv = set(mg16.morse_set(v))
        Xv = set()
        for m in Mv:
            Xv.update(adj16[m])
        Av = Xv - Mv
        badv = sorted(a for a in Av if Mv & set(adj16[a]))
        label = "p*" if cell_of(b16, Mv, P_STAR) else "0" if cell_of(b16, Mv, ORIGIN) else "?"
        out(f"   Morse set {v} (contains {label}): {len(Mv)} cells, |X| = {len(Xv)}, |A| = {len(Av)}, "
            f"cells of A whose image meets it: {badv if badv else 'none'}, annotation {list(mg16.annotations(v))}")
        r16[label] = {"vertex": v, "cells": len(Mv), "X": len(Xv), "A": len(Av), "A_cells_mapping_in": badv,
                      "annotation": list(mg16.annotations(v))}
    comps16 = LC.recurrent_components(adj16)
    origin_cell = cell_of(b16, range(N16), ORIGIN)[0]
    S16 = next(c for c in comps16 if origin_cell in c)
    named16 = {k: [cell_of(b16, S16, p) != [] for p in pts] for k, pts in LC.NAMED_POINTS.items()}
    c16 = next(i for i in range(N16) if d16[i] == 1)
    cp16 = next(i for i in range(N16) if d16[i] == 2)
    out(f"   the final cell graph has {len(comps16)} strongly connected components with a cycle, of sizes "
        f"{sorted(len(c) for c in comps16)}; the one that contains the origin cell has {len(S16)} cells, depths "
        f"{histogram(d16[c] for c in S16)}")
    out(f"   it contains C: {c16 in S16}, C': {cp16 in S16}, and the points {named16}")
    owner16 = {c: v for v in range(mg16.num_vertices()) for c in mg16.morse_set(v)}
    where16 = {}
    for name in ("a3", "s3"):
        for k, pt in enumerate(LC.NAMED_POINTS[name]):
            cells = cell_of(b16, range(N16), pt)
            where16[f"{name}[{k}]"] = [(c, d16[c], owner16.get(c)) for c in cells]
    out("   cells that contain the points of a3 and s3 (cell, depth, Morse set or None): "
        + "; ".join(f"{k}: {v}" for k, v in where16.items()))
    r16["three_cycle_cells"] = where16
    r16["component_of_origin"] = {"cells": len(S16), "depths": histogram(d16[c] for c in S16), "contains": named16,
                                  "contains_C": c16 in S16, "contains_C'": cp16 in S16}
    rec["run_16_18"] = r16
    out.json(rec)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
