"""How the Conley phase computes the indices of E1-E6, made explicit on the installed build.

For each example and each Morse set S of its run, the script reconstructs
what chomp::ConleyIndex computes, and compares the reconstruction with the
rectangles that the box map actually receives.

  1. X = cover(F(S)) on the final grid, by the rule of TreeGrid::cover, and
     A = X minus S, with the depth of every cell.
  2. The complex at d = depth(S): the cubes of TreeGrid::GridElementToCubes,
     the n cubes per axis, the cubes S' of S and A' of A.
  3. The bounds of the complex before the fix (the union of the cells of X)
     and after it (the union of the cubes), and the rectangle
     CubicalComplex::geometryOfCube assigns to each cube with either bounds.
  4. The rectangles the box map receives in the Conley phase, recorded by
     wrapping F.  These are the evaluations of ComputeConleyMorseGraph after
     the first m, where m is the number made by ComputeMorseGraph on a fresh
     Model.  The script checks that the first m coincide, in order, with
     those of ComputeMorseGraph.  It sets CMGDB_MAPGRAPH_CACHE=0 before
     CMGDB computes anything, so that the fork's builds do not end the run
     with a pass over the grid for the returned MapGraph (upstream 1.5.2
     ignores the variable and makes no such pass).  The evaluations of each
     Morse set start with the cells of S (the evaluation that yields X) and
     continue with the cubes.  Each recorded rectangle is matched with the
     rectangle of a cube from the bounds before or after the fix.
  5. The combinatorial maps on the cubes, from the images of these
     rectangles and the rule of CubicalComplex::cover (bounds_common.complex_cover):
     the map before the fix and the map after it.  For each, the script
     checks the condition F(A') cap X' subset A' that makes (X', A') an
     index pair (chomp/ConleyIndex.h), lists the cubes with empty image, and
     computes CMGDB.ComputeConleyIndex(X', A', F), chomp's own lift on the
     given combinatorial map, which uses no geometry.  It also recomputes
     the maps with closed boxes in place of half-open ones
     (bounds_common.closed_cover), a rule CMGDB does not use, to show where an
     image touches a face of a cube.
  6. On the line, the chain lift of bounds_lift1d.py for every choice it
     allows, and the affine map phi that takes each true cube onto its
     rectangle from the old bounds.

Usage:  python bounds_mechanism.py [--tag TAG] [--cases E1 ...]
Outputs: outputs/bounds_mechanism_<TAG>.txt (summary of every Morse set and
details for the Morse sets of the fixed points (1), (11/16) and (0, 1)) and
outputs/bounds_mechanism_<TAG>.json (everything, read by bounds_figures.py).
Runs on every build listed in bounds_common.py.  Runtime: about 1 s.
"""

import os
import sys
from pathlib import Path

os.environ["CMGDB_MAPGRAPH_CACHE"] = "0"     # before CMGDB computes anything
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402
import bounds_lift1d as L  # noqa: E402

TOL = 1e-12          # matching recorded rectangles with predicted ones
DETAIL = {"E1": [(1.0,)], "E2": [(1.0,)], "E3": [(C._P_LOG,)],
          "E4": [(0.0, 1.0)], "E5": [(0.0, 1.0)], "E6": [(0.0, 1.0)]}


def qname(q):
    return f"c{q[0]}" if len(q) == 1 else "(" + ", ".join(str(t) for t in q) + ")"


def map_text(M):
    return ", ".join(qname(q) + " -> {" + ", ".join(qname(t) for t in M[q]) + "}" for q in sorted(M))


def phase_logs(c):
    """Map evaluations of ComputeMorseGraph and ComputeConleyMorseGraph."""
    import CMGDB
    F = C.case_F(c)
    rec_m = C.Recorder(F)
    CMGDB.ComputeMorseGraph(C.make_model(c, rec_m))
    rec_c = C.Recorder(F)
    error = None
    try:
        mg, mapg = CMGDB.ComputeConleyMorseGraph(C.make_model(c, rec_c))
    except Exception as e:  # recorded, not hidden
        error = f"{type(e).__name__}: {e}"
        mg, mapg = CMGDB.ComputeMorseGraph(C.make_model(c, F))
    m = len(rec_m.log)
    prefix_ok = rec_c.log[:m] == rec_m.log
    return mg, mapg, error, rec_c.log[m:], prefix_ok, m


def segment(log, sets):
    """Split the Conley-phase evaluations by Morse set: for each set, in the
    order v = 0, 1, ..., the block of evaluations on its cells and the
    evaluations that follow it up to the next such block."""
    pos, starts = 0, []
    for boxes in sets:
        want = sorted(tuple(b) for b in boxes)
        k = len(want)
        j = pos
        while j + k <= len(log) and sorted(tuple(r) for r in log[j:j + k]) != want:
            j += 1
        if j + k > len(log):
            break
        starts.append(j)
        pos = j + k
    segs = {}
    for v, j in enumerate(starts):
        k = len(sets[v])
        end = starts[v + 1] if v + 1 < len(starts) else len(log)
        segs[v] = dict(start=j, S_block=log[j:j + k], cubes=log[j + k:end])
    return segs


def analyze(c, grid, gidx, S, F):
    """Steps 1-3 and 5 of the module docstring for the Morse set with cell
    indices S."""
    import CMGDB
    lb, ub, D = c["lb"], c["ub"], c["dim"]
    dS = max(gidx[i][0] for i in S)
    X = sorted({j for i in S for j in C.tree_cover(F(grid[i]), gidx, lb, ub)})
    Sset = set(S)
    A = [i for i in X if i not in Sset]
    cubes_of = {i: C.cubes_of_cell(gidx[i][0], gidx[i][1], dS, D) for i in X}
    allc = sorted({q for i in X for q in cubes_of[i]})
    mn = [min(q[d] for q in allc) for d in range(D)]
    mx = [max(q[d] for q in allc) for d in range(D)]
    n = [mx[d] - mn[d] + 1 for d in range(D)]
    off = lambda q: tuple(q[d] - mn[d] for d in range(D))
    K = sorted({off(q) for q in allc})
    Kset = set(K)
    S_c = sorted({off(q) for i in S for q in cubes_of[i]})
    A_c = sorted({off(q) for i in A for q in cubes_of[i]})
    old = C.hull([grid[i] for i in X])
    tree_box = {off(q): C.tree_node_geometry(dS, list(q), lb, ub) for q in allc}
    true = C.hull([tree_box[q] for q in K])
    rect_old = {q: C.cube_geometry(q, old, n) for q in K}
    rect_true = {q: C.cube_geometry(q, true, n) for q in K}
    maps, checks, conley = {}, {}, {}
    for label, bounds, rects in (("before", old, rect_old), ("after", true, rect_true)):
        imgs = {q: F(rects[q]) for q in K}
        maps[label] = {q: C.complex_cover(imgs[q], bounds, n, Kset) for q in K}
        maps[label + "_closed"] = {q: C.closed_cover(imgs[q], bounds, n, K) for q in K}
        maps[label + "_images"] = imgs
    for label in ("before", "after", "before_closed", "after_closed"):
        M = maps[label]
        into_S = [(a, [s for s in M[a] if s in set(S_c)]) for a in A_c]
        into_S = [(a, s) for a, s in into_S if s]
        checks[label] = dict(exit_cubes_into_S=into_S, empty=[q for q in K if not M[q]],
                             index_pair=not into_S)
        Xc = [C.flat_index(q, n) for q in K]
        Ac = [C.flat_index(q, n) for q in A_c]
        Fm = {C.flat_index(q, n): [C.flat_index(t, n) for t in M[q]] for q in K}
        try:
            conley[label] = [str(s) for s in CMGDB.ComputeConleyIndex(Xc, Ac, list(n), [False] * D, Fm, True)]
        except Exception as e:  # recorded, not hidden
            conley[label] = f"{type(e).__name__}: {e}"
    lift = None
    if D == 1:
        lift = {}
        for label in ("before", "after"):
            M = {q[0]: [t[0] for t in maps[label][q]] for q in K}
            rep = L.lift_outcomes([q[0] for q in K], [q[0] for q in A_c], M)
            rep["outcomes"] = [[dict(coordinates=list(o) if isinstance(o, tuple) else None,
                                     annotation=L.outcome_text(o) if isinstance(o, tuple) else None,
                                     status="ok" if isinstance(o, tuple) else o,
                                     vertices=[ch[0] if len(ch) == 1 else ch for ch in ws])
                                for o, ws in lst] for lst in rep["outcomes"]]
            lift[label] = rep
    phi = [dict(slope=(old[D + d] - old[d]) / (true[D + d] - true[d]), origin_true=true[d], origin_old=old[d])
           for d in range(D)]
    return dict(depth_S=dS, S=S, X=X, A=A, X_depths=[gidx[i][0] for i in X],
                deeper=[i for i in X if gidx[i][0] > dS], cubes_of={i: [off(q) for q in cubes_of[i]] for i in X},
                n=n, mincube=mn, K=K, S_cubes=S_c, A_cubes=A_c, bounds_before=old, bounds_after=true,
                short_axes=[d for d in range(D) if old[d] != true[d] or old[D + d] != true[D + d]],
                rect_before=rect_old, rect_after=rect_true, tree_box=tree_box,
                maps=maps, checks=checks, conley_index=conley, lift=lift, phi=phi)


def match_recorded(seg, an, F, c):
    """Step 4: match the recorded cube evaluations with predicted rectangles,
    and rebuild the map this build used from the recorded rectangles."""
    rows, unmatched, dev_max = [], [], 0.0
    used = {}
    for r in seg["cubes"]:
        hit = None
        for label in ("before", "after"):
            for q, g in an["rect_" + label].items():
                dev = max(abs(a - b) for a, b in zip(r, g))
                if dev <= TOL:
                    hit = (label, q, dev)
                    break
            if hit:
                break
        if hit is None:
            unmatched.append(r)
            continue
        label, q, dev = hit
        dev_max = max(dev_max, dev)
        img = F(r)
        bounds = an["bounds_" + label]
        cov = C.complex_cover(img, bounds, an["n"], set(an["K"]))
        rows.append(dict(rect=r, cube=q, bounds=label, deviation=dev, image=img, cover=cov,
                         grid_box=C.is_subdivision_box(r, c["lb"], c["ub"])))
        used.setdefault(q, cov)
    labels = sorted({row["bounds"] for row in rows})
    if an["bounds_before"] == an["bounds_after"]:
        labels = ["before and after"] if rows else []
    same = None
    if len(labels) == 1 and used:
        key = "before" if labels[0] == "before and after" else labels[0]
        same = labels[0] if all(used[q] == an["maps"][key][q] for q in used) else "neither"
    distinct = len({tuple(r) for r in seg["cubes"]})
    return dict(S_block=seg["S_block"], evaluations=rows, unmatched=unmatched, max_deviation=dev_max,
                bounds_used=labels, cubes_evaluated=sorted(used), map_used=same,
                n_cube_evaluations=len(seg["cubes"]), n_distinct=distinct)


def jsonable(an):
    """Tuples as lists and maps as lists of pairs, for json."""
    out = {}
    for k, v in an.items():
        if k in ("rect_before", "rect_after", "tree_box"):
            out[k] = [[list(q), r] for q, r in sorted(v.items())]
        elif k == "maps":
            out[k] = {lab: [[list(q), [list(t) for t in M[q]] if not lab.endswith("_images") else M[q]]
                            for q in sorted(M)] for lab, M in v.items()}
        elif k == "checks":
            out[k] = {lab: dict(exit_cubes_into_S=[[list(a), [list(s) for s in ss]] for a, ss in ch["exit_cubes_into_S"]],
                                empty=[list(q) for q in ch["empty"]], index_pair=ch["index_pair"])
                      for lab, ch in v.items()}
        elif k == "cubes_of":
            out[k] = [[i, [list(q) for q in qs]] for i, qs in sorted(v.items())]
        elif k in ("K", "S_cubes", "A_cubes"):
            out[k] = [list(q) for q in v]
        else:
            out[k] = v
    return out


def detail(out, c, grid, gidx, ms, an, rec, F):
    """Detailed listing for one Morse set."""
    D = c["dim"]
    nd = 9
    S, X, A = an["S"], set(an["X"]), set(an["A"])
    out(f"    S: {len(S)} cell(s) of depth {an['depth_S']}: " +
        "; ".join(f"{C.fmt_box(grid[i], nd)}, F(cell) = {C.fmt_box(F(grid[i]), nd)}" for i in S))
    # cells near S: those that meet the true extent of the complex widened by one cube
    tb = an["bounds_after"]
    w = [(tb[D + d] - tb[d]) / an["n"][d] for d in range(D)]
    win = [tb[d] - w[d] for d in range(D)] + [tb[D + d] + w[d] for d in range(D)]
    near = [i for i, b in enumerate(grid)
            if all(b[d] < win[D + d] and b[D + d] > win[d] for d in range(D))]
    out(f"    grid cells that meet {C.fmt_box(win, 6)} (depth, and S or A = X minus S):")
    for i in sorted(near, key=lambda i: [grid[i][d] for d in reversed(range(D))]):
        tag = "S" if i in set(S) else ("A" if i in A else "")
        cubes = ", ".join(qname(q) for q in an["cubes_of"][i]) if i in X else ""
        out(f"      {C.fmt_box(grid[i], nd)}  depth {gidx[i][0]:2d}  {tag:1s}" +
            (f"  enters the complex as {cubes}" if cubes else ""))
    out(f"    X = cover(F(S)): {len(X)} cells; A = X minus S: {len(A)} cells; "
        f"{len(an['deeper'])} cells of X deeper than depth(S) = {an['depth_S']}")
    out(f"    complex at depth {an['depth_S']}: n = {an['n']} cubes per axis; S' = "
        f"{[qname(q) for q in an['S_cubes']]}, A' = {[qname(q) for q in an['A_cubes']]}")
    ob, tb = an["bounds_before"], an["bounds_after"]
    out(f"    bounds before the fix (union of the cells of X): {C.fmt_box(ob, nd)}")
    out(f"    bounds after the fix (union of the cubes):        {C.fmt_box(tb, nd)}")
    out("    cube widths before / after: " + ", ".join(
        f"{(ob[D + d] - ob[d]) / an['n'][d]:.9f} / {(tb[D + d] - tb[d]) / an['n'][d]:.9f}" for d in range(D)))
    out("    cube: true box | box from the bounds before the fix | its image under F")
    for q in an["K"]:
        tag = "S'" if q in an["S_cubes"] else "A'"
        out(f"      {qname(q)} ({tag}): {C.fmt_box(an['rect_after'][q], nd)} | "
            f"{C.fmt_box(an['rect_before'][q], nd)} | {C.fmt_box(an['maps']['before_images'][q], nd)}")
    out("    images of the true boxes: " + "; ".join(
        f"F({qname(q)}) = {C.fmt_box(an['maps']['after_images'][q], nd)}" for q in an["K"]))
    if rec is not None:
        out(f"    recorded in this build: F evaluated on the {len(rec['S_block'])} cell(s) of S, then on "
            f"{len(rec['evaluations'])} rectangle(s) of cubes (with repeats), {len(rec['unmatched'])} unmatched:")
        seen = set()
        for row in rec["evaluations"]:
            key = (tuple(row["rect"]), row["cube"])
            if key in seen:
                continue
            seen.add(key)
            k = sum(1 for r2 in rec["evaluations"] if tuple(r2["rect"]) == key[0])
            out(f"      {C.fmt_box(row['rect'], nd)} = box of {qname(row['cube'])} from the bounds "
                f"{row['bounds']} the fix (deviation {row['deviation']:.1e}); box of the bisection tree: "
                f"{row['grid_box']}; evaluations {k}; image {C.fmt_box(row['image'], nd)}; "
                f"cover {[qname(t) for t in row['cover']]}")
        out(f"    map this build used, rebuilt from the recorded boxes: on the {len(rec['cubes_evaluated'])} "
            f"evaluated cubes, the map {rec['map_used']} the fix")
    for label, text in (("before", "before the fix"), ("after", "after the fix")):
        ch = an["checks"][label]
        viol = "; ".join(f"{qname(a)} in A' maps to {[qname(s) for s in ss]} in S'" for a, ss in ch["exit_cubes_into_S"])
        out(f"    map on the cubes {text}: {map_text(an['maps'][label])}")
        out(f"      F(A') cap X' subset A': {'holds' if ch['index_pair'] else 'FAILS, ' + viol}; "
            f"cubes with empty image: {[qname(q) for q in ch['empty']] or 'none'}; "
            f"ComputeConleyIndex: {an['conley_index'][label]}")
    for label, text in (("before_closed", "before the fix"), ("after_closed", "after the fix")):
        ch = an["checks"][label]
        viol = "; ".join(f"{qname(a)} -> {[qname(s) for s in ss]}" for a, ss in ch["exit_cubes_into_S"])
        out(f"    with closed boxes instead (not CMGDB's rule), {text}: {map_text(an['maps'][label])}; "
            f"index pair: {'yes' if ch['index_pair'] else 'no, ' + viol}; ComputeConleyIndex: {an['conley_index'][label]}")
    for d, ph in enumerate(an["phi"]):
        if an["bounds_before"][d] != an["bounds_after"][d] or an["bounds_before"][D + d] != an["bounds_after"][D + d]:
            p = ms["point"][d]
            pinv = ph["origin_true"] + (p - ph["origin_old"]) / ph["slope"]
            cube = next((q for q in an["K"] if an["rect_after"][q][d] <= pinv <= an["rect_after"][q][D + d]), None)
            out(f"    axis {d}: phi(x) = {ph['origin_old']:.9f} + {ph['slope']:.9f} (x - {ph['origin_true']:.9f}) "
                f"takes each true cube onto its box from the bounds before the fix; "
                f"phi^-1({p:g}) = {pinv:.9f}, in cube {qname(cube) if cube else 'none'} along this axis")
    if an["lift"]:
        t0, h = an["bounds_after"][0], (an["bounds_after"][1] - an["bounds_after"][0]) / an["n"][0]
        verts = ", ".join(f"v{k} = {t0 + k * h:g}" for k in range(an["n"][0] + 1))
        out(f"    chain lift (bounds_lift1d.py); vertex v_k is the left end of c_k, in the true geometry {verts}")
        for label, text in (("before", "before the fix"), ("after", "after the fix")):
            rep = an["lift"][label]
            cyc = " ; ".join(" + ".join(f"{v} {qname((k,))}" for k, v in z) for z in rep["cycles"])
            out(f"      map {text}: dim H_0(X', A') = {rep['H0_dim']}, dim H_1(X', A') = {rep['H1_dim']}, "
                f"H_1 spanned by z = {cyc or 'none'}; preboundaries "
                f"{'unique' if rep['kernel_max'] == 0 else 'not unique (kernel dimension up to %d)' % rep['kernel_max']}")
            for lst in rep["outcomes"]:
                for o in lst:
                    ws = ", ".join(f"v{k}" for k in o["vertices"])
                    res = (f"image {o['coordinates'][0]} z, annotation in degree 1 '{o['annotation']}'"
                           if o["status"] == "ok" else o["status"])
                    out(f"        w = {ws}: {res}")


def main():
    def extra(p):
        p.add_argument("--cases", nargs="*", default=list(C.CASES))
    args = C.parse_args(__doc__, extra)
    out = C.Output("bounds_mechanism", args.tag)
    out(*C.header_lines(__file__, args.tag, env=False))
    out(f"CMGDB_MAPGRAPH_CACHE = {os.environ.get('CMGDB_MAPGRAPH_CACHE')}", "")
    record = dict(script=Path(__file__).name, cmgdb_version=C.cmgdb_version(), tag=args.tag, cases={})
    for name in args.cases:
        c = C.CASES[name]
        F = C.case_F(c)
        lb, ub = c["lb"], c["ub"]
        mg, mapg, error, clog, prefix_ok, m = phase_logs(c)
        grid = C.grid_boxes(mg, mapg)
        gidx = [C.cell_index(b, lb, ub) for b in grid]
        index = {tuple(b): i for i, b in enumerate(grid)}
        sets = C.morse_sets(mg)
        segs = segment(clog, sets)
        out(C.case_title(c))
        out("  ComputeConleyMorseGraph: " + (f"raised {error}" if error else "returned") +
            f"; {m} evaluations of ComputeMorseGraph, the same in the same order at the start of "
            f"ComputeConleyMorseGraph: {prefix_ok}; {len(clog)} evaluations after them (the Conley phase), "
            f"{sum(not C.is_subdivision_box(r, lb, ub) for r in clog)} of them not boxes of the bisection tree")
        rc = dict(title=C.case_title(c), error=error, prefix_ok=prefix_ok, n_morse_phase=m,
                  conley_log=clog, grid=grid, depths=[g[0] for g in gidx], morse_sets=[])
        for v, boxes in enumerate(sets):
            S = sorted(index[tuple(b)] for b in boxes)
            an = analyze(c, grid, gidx, S, F)
            ann = None if error else [str(s) for s in mg.annotations(v)]
            pts = [p for p in c["fixed"] if any(C.contains(b, p) for b in boxes)]
            rec = match_recorded(segs[v], an, F, c) if v in segs else None
            ch_b, ch_a = an["checks"]["before"], an["checks"]["after"]
            out(f"  Morse set {v}{' (fixed point ' + C.fmt_point(pts[0]) + ')' if pts else ''}: annotation "
                f"{ann if ann is not None else '(none, exception)'}; S: {len(S)} cell(s) of depth {an['depth_S']}; "
                f"X: {len(an['X'])} cells, {len(an['deeper'])} deeper than S; n = {an['n']}; "
                f"axes with bounds short before the fix: {an['short_axes'] or 'none'}")
            if rec is None:
                out("    boxes the map received for this set: none recorded (the run stopped before it)")
            else:
                src = " and ".join(rec["bounds_used"]) or "no cube"
                out(f"    boxes the map received for the cubes: from the bounds {src} the fix, "
                    f"{len(rec['cubes_evaluated'])} of {len(an['K'])} cubes evaluated "
                    f"({rec['n_cube_evaluations']} evaluations, {rec['n_distinct']} distinct), {len(rec['unmatched'])} "
                    f"rectangles unmatched, largest deviation {rec['max_deviation']:.1e}; on the evaluated cubes "
                    f"they yield the map {rec['map_used']} the fix")
            out(f"    before the fix: index pair {ch_b['index_pair']}"
                f"{' (' + '; '.join(qname(a) + ' -> ' + str([qname(s) for s in ss]) for a, ss in ch_b['exit_cubes_into_S']) + ')' if not ch_b['index_pair'] else ''}"
                f", empty images {[qname(q) for q in ch_b['empty']] or 'none'}, ComputeConleyIndex {an['conley_index']['before']}"
                f"; closed boxes: index pair {an['checks']['before_closed']['index_pair']}, "
                f"ComputeConleyIndex {an['conley_index']['before_closed']}")
            out(f"    after the fix:  index pair {ch_a['index_pair']}, empty images "
                f"{[qname(q) for q in ch_a['empty']] or 'none'}, ComputeConleyIndex {an['conley_index']['after']}"
                f"; closed boxes: index pair {an['checks']['after_closed']['index_pair']}, "
                f"ComputeConleyIndex {an['conley_index']['after_closed']}")
            ms = dict(v=v, boxes=boxes, annotation=ann, fixed=[list(p) for p in pts],
                      analysis=jsonable(an), recorded=rec, point=list(pts[0]) if pts else None)
            if pts and any(tuple(p) == tuple(q) for p in pts for q in DETAIL.get(name, [])):
                out(f"  Details for Morse set {v}:")
                detail(out, c, grid, gidx, ms, an, rec, F)
            rc["morse_sets"].append(ms)
        record["cases"][name] = rc
        out("")
    out.json(record)
    out.close()


if __name__ == "__main__":
    main()
