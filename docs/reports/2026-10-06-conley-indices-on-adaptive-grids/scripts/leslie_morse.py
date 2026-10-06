"""Morse decompositions of the Leslie map on Q = [-0.001, 90] x [-0.001, 70].

Runs ComputeConleyMorseGraph for

  corner 12/14, corner 16/18           Model(min, max, lower, upper, BoxMap(f, B))
  corner uniform 12, 14, 16            Model(d, d, d, 10000, lower, upper, BoxMap(f, B))
  corner uniform 18                    only with --heavy
  hull 12/14, hull 16/18               box map: the smallest box containing f(B)
  padded 12/14, padded 16/18           BoxMap(f, B, padding=True)

and reports, for every run, the grid (number of cells, depths), and for every
Morse set M: its cells, depths and bounding box, the invariant sets it
contains (0, p*, the points of a3 and s3, and how many of 2000 points of
Gamma), its annotation, the check of the pair (X, A) = (cover F(M),
cover F(M) minus M) (the cells of A whose image meets M = X minus A), and the
value of ComputeConleyIndexForCells (or the error it raises).  It also lists
the Morse graph edges, the strongly connected components of the final cell
graph that contain a cycle, and the cells of depth < min on cycles.

The JSON output holds SHA-256 digests of the grid and of every Morse set.
With --boxes it also holds the boxes that the figures need: the grids of the
adaptive depth-16 runs and the Morse sets of the uniform depth-16 run.

Output: outputs/leslie_morse_<tag>.txt and outputs/leslie_morse_<tag>.json.
Runs on every build that has ComputeConleyMorseGraph and
ComputeConleyIndexForCells (master 96642f2, upstream 1.5.2, fork 1.3.3+fork.6,
merge 0e21503).  On the Apple silicon machine of the stored outputs it takes
about 3 s, and about 5 s with --heavy (uniform depth 18, 262144 cells).
numpy only.

usage: python leslie_morse.py --tag TAG [--heavy] [--boxes]
"""

import hashlib
import json
import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
from leslie_common import NAMED_POINTS, fmt_box, histogram, depth_of


RUNS = [
    # label, box map, min, max, init (None: Model(min, max, lower, upper, F))
    ("corner 12/14", "corner", 12, 14, None),
    ("corner 16/18", "corner", 16, 18, None),
    ("corner uniform 12", "corner", 12, 12, 12),
    ("corner uniform 14", "corner", 14, 14, 14),
    ("corner uniform 16", "corner", 16, 16, 16),
    ("corner uniform 18", "corner", 18, 18, 18),
    ("hull 12/14", "hull", 12, 14, None),
    ("hull 16/18", "hull", 16, 18, None),
    ("padded 12/14", "padded", 12, 14, None),
    ("padded 16/18", "padded", 16, 18, None),
]
HEAVY = {"corner uniform 18"}
FIGURE_RUNS = {"corner 16/18", "corner uniform 16", "hull 16/18", "padded 16/18"}


def digest(boxes):
    """SHA-256 (first 16 hex digits) of a list of boxes, floats written exactly."""
    return hashlib.sha256(json.dumps([[float(v).hex() for v in b] for b in boxes]).encode()).hexdigest()[:16]


def points_in_union(boxes, cells, pts):
    """Boolean array: which of the points lie in the union of the closed cells."""
    B = np.asarray(boxes, float)[np.asarray(cells, int)]
    P = np.asarray(pts, float)
    hit = np.zeros(len(P), bool)
    for k in range(0, len(B), 4096):
        b = B[k:k + 4096]
        hit |= ((b[None, :, 0] <= P[:, None, 0]) & (P[:, None, 0] <= b[None, :, 2]) &
                (b[None, :, 1] <= P[:, None, 1]) & (P[:, None, 1] <= b[None, :, 3])).any(axis=1)
    return hit


def contents(boxes, cells, gamma):
    """Named points (0, p*, a3[k], s3[k]) in the union of the cells, and the
    number of the Gamma points in it."""
    found = []
    for name, pts in NAMED_POINTS.items():
        hit = [k for k, h in enumerate(points_in_union(boxes, cells, pts)) if h]
        if len(pts) == 1:
            if hit:
                found.append(name)
        elif len(hit) == len(pts):
            found.append(name)
        elif hit:
            found.append(f"{name}{hit}")
    ng = int(points_in_union(boxes, cells, gamma).sum())
    return found, ng


def label_of(found, ng, ngamma):
    names = [n for n in found if n in ("p*", "a3", "s3", "0")]
    if ng == ngamma:
        names.append("Gamma")
    elif ng:
        names.append(f"Gamma({ng}/{ngamma})")
    return ", ".join(names) if names else "none of these"


def run(label, box_map, smin, smax, init, gamma, out, keep_boxes):
    import CMGDB
    model = LC.leslie_model(smin, smax, init, box_map)
    mg, gr = CMGDB.ComputeConleyMorseGraph(model)
    N = gr.num_vertices()
    boxes = [list(mg.phase_space_box(i)) for i in range(N)]
    boxes_np = np.asarray(boxes, float)
    depths = [depth_of(b) for b in boxes]
    adj = [sorted(gr.adjacencies(i)) for i in range(N)]
    nv = mg.num_vertices()
    out("", f"== {label}: Model({smin}, {smax}" + (f", {init}, {LC.LIMIT}" if init is not None else "") +
        f", lower, upper, {box_map} box map)")
    out(f"   final grid: {N} cells, depths {histogram(depths)}")
    owner = {}
    for v in range(nv):
        for c in mg.morse_set(v):
            owner[c] = v
    sets = []
    for v in range(nv):
        cells = sorted(mg.morse_set(v))
        found, ng = contents(boxes_np, cells, gamma)
        ann = list(mg.annotations(v))
        X = set()
        for c in cells:
            X.update(adj[c])
        Ms = set(cells)
        A = X - Ms
        bad = sorted(a for a in A if any(t in Ms for t in adj[a]))
        try:
            idx = list(CMGDB.ComputeConleyIndexForCells(model, mg, cells))
        except ValueError as e:
            idx = "ValueError: " + str(e)
        except Exception as e:               # report any other error, do not stop
            idx = f"{type(e).__name__}: {e}"
        bb = LC.bounding_box([boxes[c] for c in cells])
        sets.append({"vertex": v, "num_cells": len(cells), "depths": histogram(depths[c] for c in cells),
                     "bbox": bb, "contains": found, "gamma_points": ng, "label": label_of(found, ng, len(gamma)),
                     "annotation": ann, "X": len(X), "A": len(A), "A_depths": histogram(depths[a] for a in A),
                     "A_cells_mapping_into_M": bad, "index_for_cells": idx,
                     "digest": digest(sorted(boxes[c] for c in cells))})
        if keep_boxes and label in FIGURE_RUNS:
            if init is None:
                sets[-1]["cells"] = cells
            else:
                sets[-1]["boxes"] = [[round(v, 6) for v in boxes[c]] for c in cells]
        out(f"   Morse set {v}: {len(cells)} cells, depths {histogram(depths[c] for c in cells)}, bounding box {fmt_box(bb, 3)}")
        out(f"      contains: {label_of(found, ng, len(gamma))}   annotation {ann}")
        out(f"      pair: |X| = {len(X)}, |A| = {len(A)} (depths {histogram(depths[a] for a in A)}); "
            f"cells of A whose image meets M: {bad if bad else 'none'}")
        out(f"      ComputeConleyIndexForCells: {idx if isinstance(idx, list) else idx[:160] + ('...' if len(idx) > 160 else '')}")
    edges = sorted(tuple(e) for e in mg.edges())
    edges_un = sorted(tuple(e) for e in mg.edges_unreduced())
    out(f"   Morse graph edges (reduced): {edges}")
    out(f"   Morse graph edges (unreduced): {edges_un}")
    paths = cell_paths(adj, owner, nv)
    closure = transitive_closure(edges_un, nv)
    out(f"   pairs (u, v) of Morse sets with a path of cells from u to v in the final graph: {paths}")
    out(f"      Morse graph pairs (transitive closure) without such a path: {sorted(closure - set(paths)) or 'none'};"
        f"  such paths without a Morse graph pair: {sorted(set(paths) - closure) or 'none'}")
    rec = recurrent_report(boxes_np, depths, adj, owner, nv, smin, gamma, out)
    record = {"label": label, "box_map": box_map, "min": smin, "max": smax, "init": init,
              "num_cells": N, "depths": histogram(depths), "digest": digest(boxes), "morse_sets": sets,
              "edges": [list(e) for e in edges], "edges_unreduced": [list(e) for e in edges_un],
              "cell_paths": [list(e) for e in paths]}
    record.update(rec)
    if keep_boxes and label in FIGURE_RUNS and init is None:
        record["boxes"] = [[round(v, 6) for v in b] for b in boxes]
    return record


def cell_paths(adj, owner, nv):
    """Pairs (u, v), u != v, of Morse sets such that a path in the final cell
    graph leads from a cell of u to a cell of v."""
    pairs = []
    for u in range(nv):
        seen = bytearray(len(adj))
        frontier = [c for c, w in owner.items() if w == u]
        for c in frontier:
            seen[c] = 1
        while frontier:
            nxt = []
            for c in frontier:
                for t in adj[c]:
                    if not seen[t]:
                        seen[t] = 1
                        nxt.append(t)
            frontier = nxt
        reached = sorted({w for c, w in owner.items() if seen[c] and w != u})
        pairs += [(u, v) for v in reached]
    return pairs


def transitive_closure(edges, nv):
    reach = {u: {v for a, v in edges if a == u} for u in range(nv)}
    changed = True
    while changed:
        changed = False
        for u in range(nv):
            extra = set().union(*(reach[v] for v in reach[u])) - reach[u] if reach[u] else set()
            if extra:
                reach[u] |= extra
                changed = True
    return {(u, v) for u in range(nv) for v in reach[u] if u != v}


def recurrent_report(boxes, depths, adj, owner, nv, smin, gamma, out):
    comps = LC.recurrent_components(adj)
    morse = {}
    for c, v in owner.items():
        morse.setdefault(v, set()).add(c)
    equal = sorted(map(tuple, comps)) == sorted(tuple(sorted(s)) for s in morse.values())
    coarse_on_cycles = sorted(c for comp in comps for c in comp if depths[c] < smin)
    out(f"   strongly connected components of the final cell graph that contain a cycle: {len(comps)}")
    rows = []
    for comp in sorted(comps, key=len, reverse=True):
        found, ng = contents(boxes, comp, gamma)
        inside = sorted(v for v, s in morse.items() if s <= set(comp))
        coarse = [c for c in comp if depths[c] < smin]
        rows.append({"num_cells": len(comp), "depths": histogram(depths[c] for c in comp),
                     "morse_sets_inside": inside, "label": label_of(found, ng, len(gamma)),
                     "cells_of_depth_below_min": len(coarse), "digest": digest(sorted(boxes[c].tolist() for c in comp))})
        out(f"      {len(comp)} cells, depths {histogram(depths[c] for c in comp)}, Morse sets inside {inside}, "
            f"contains {label_of(found, ng, len(gamma))}"
            + (f"; cells of depth < {smin}: {len(coarse)}" if coarse else ""))
    out(f"   these components are the Morse sets: {equal};  cells of depth < {smin} on a cycle: "
        f"{len(coarse_on_cycles)}" + (f" (depths {histogram(depths[c] for c in coarse_on_cycles)})" if coarse_on_cycles else ""))
    return {"recurrent_components": rows, "components_equal_morse_sets": equal,
            "cells_below_min_on_cycles": len(coarse_on_cycles)}


def main():
    def extra(p):
        p.add_argument("--heavy", action="store_true", help="also run the uniform depth-18 decomposition")
        p.add_argument("--boxes", action="store_true", help="store the boxes that the figures need")
    args = LC.parse_args(__doc__, extra)
    out = LC.Output("leslie_morse", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    gamma = LC.gamma_orbit(2000, step=50)
    out(f"Gamma: 2000 points of the orbit of (30, 20) after 100000 steps (every 50th point)")
    out(f"named points: 0, p* = ({LC.P_STAR[0]:.6f}, {LC.P_STAR[1]:.6f}), "
        f"a3 = {[tuple(round(v, 6) for v in q) for q in LC.A3]}, s3 = {[tuple(round(v, 6) for v in q) for q in LC.S3]}")
    records = []
    for label, box_map, smin, smax, init in RUNS:
        if label in HEAVY and not args.heavy:
            out("", f"== {label}: skipped (use --heavy)")
            continue
        records.append(run(label, box_map, smin, smax, init, gamma, out, args.boxes))
    out.json({"cmgdb_version": LC.cmgdb_version(), "tag": args.tag, "runs": records})
    out.close()


if __name__ == "__main__":
    sys.exit(main())
