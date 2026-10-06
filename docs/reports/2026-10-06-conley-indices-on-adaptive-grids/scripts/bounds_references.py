"""The linearization and the uniform runs of E1-E6, with which the annotations are compared.

Linearization.  For each fixed point p of an example, the script prints the
annotation that the linearization predicts (bounds_common.theory_annotation)
and checks, on the Morse set S that contains p, the conditions under which
the Conley index of S is the index of {p}:

  (a) on each axis i, |f_i'| > 1 on the whole interval [a_i, b_i] of the
      hull of |S| (axis i expanding) or |f_i'| < 1 there (axis i
      contracting).  Both maps have monotone derivatives on these intervals
      (2 / (2 - x)^2 is increasing on [0, 2) and 3.2 (1 - 2x) is affine), so
      the extremes of |f_i'| are taken at the end points.  Since f is a
      product of maps of the line, (a) implies Inv(|S|) = {p}: on an expanding
      axis the mean value theorem yields |f^n(x)_i - p_i| >= m^n |x_i - p_i|
      with m > 1 as long as the orbit stays in |S|, and on a contracting axis
      the same holds for backward orbits.
  (b) p lies in the interior of |S| relative to the phase space: on each
      axis a_i < p_i < b_i, or p_i is an end of the domain.  In the second
      case the face {x_i = p_i} is invariant (f_i(p_i) = p_i), f_i is
      increasing, so f maps the phase-space side of the face into itself,
      and axis i is contracting.  The boundary then cuts only a stable
      direction, the pair ([p_i, p_i + e], empty set) in that factor has the
      homology of a point, and the index of {p} for the map on the phase
      space is the index in R^D.  If the face cut an unstable direction,
      the index would be trivial.

Uniform grids.  Model(d, d, d) for the coarse and the fine depth of each
example (subdiv_min and subdiv_max of its adaptive model).  Every cell then
has depth d, no cell of X is deeper than S, and the bounds of the complex are
exact on every build.  The script prints the annotations of the Morse sets
that contain each fixed point, the number of map evaluations that are not
boxes of the bisection tree (0 expected), and, for the 1-D examples, all
Morse sets.

Usage:  python bounds_references.py [--tag TAG] [--cases E1 ...]
Outputs: outputs/bounds_references_<TAG>.txt and .json.
Runs on every build listed in bounds_common.py.  Runtime: about 1 s.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402


def check_conditions(c, p, boxes):
    """Conditions (a) and (b) of the module docstring for the Morse set with
    the given boxes and the fixed point p."""
    m = C.MAPS[c["map"]]
    f, df = m["f"], m["df"]
    D = c["dim"]
    h = C.hull(boxes)
    axes = []
    ok = True
    for i in range(D):
        a, b = h[i], h[D + i]
        d_lo, d_hi = abs(df(a)), abs(df(b))
        lo, hi = min(d_lo, d_hi), max(d_lo, d_hi)
        kind = "expanding" if lo > 1.0 else ("contracting" if hi < 1.0 else "neither")
        interior = a < p[i] < b
        at_face = (p[i] == c["lb"][i] == a) or (p[i] == c["ub"][i] == b)
        face_ok = None
        if at_face:
            face_ok = (f(p[i]) == p[i]) and min(df(a), df(b)) > 0 and kind == "contracting"
        axis_ok = kind != "neither" and (interior or (at_face and face_ok))
        ok = ok and axis_ok
        axes.append(dict(interval=[a, b], abs_df=[lo, hi], kind=kind, interior=interior,
                         at_face=at_face, face_invariant_and_stable=face_ok, ok=axis_ok))
    return dict(hull=h, axes=axes, applies=ok)


def describe(chk):
    parts = []
    for i, ax in enumerate(chk["axes"]):
        pos = "p_i inside" if ax["interior"] else ("p_i on a face of the domain" if ax["at_face"] else "p_i on the boundary of |S|")
        s = (f"axis {i}: [{ax['interval'][0]:.6f}, {ax['interval'][1]:.6f}], |f'| in "
             f"[{ax['abs_df'][0]:.4f}, {ax['abs_df'][1]:.4f}] ({ax['kind']}), {pos}")
        if ax["at_face"]:
            s += ", face invariant and axis contracting" if ax["face_invariant_and_stable"] else ", face condition FAILS"
        parts.append(s)
    return parts


def run(c, subdiv, conley=True):
    import CMGDB
    F = C.case_F(c)
    rec = C.Recorder(F)
    model = C.make_model(c, rec, subdiv)
    if conley:
        mg, mapg = CMGDB.ComputeConleyMorseGraph(model)
    else:
        mg, mapg = CMGDB.ComputeMorseGraph(model)
    lb, ub = c["lb"], c["ub"]
    grid = C.grid_boxes(mg, mapg)
    depths = sorted({C.cell_index(b, lb, ub)[0] for b in grid})
    sets = C.morse_sets(mg)
    anns = [[str(s) for s in mg.annotations(v)] for v in range(len(sets))] if conley else None
    off = sum(not C.is_subdivision_box(r, lb, ub) for r in rec.log)
    return dict(subdiv=list(subdiv), n_cells=len(grid), depths=depths, n_morse_sets=len(sets),
                morse_sets=sets, annotations=anns, n_evaluations=len(rec.log), n_off_grid=off)


def main():
    def extra(p):
        p.add_argument("--cases", nargs="*", default=list(C.CASES))
    args = C.parse_args(__doc__, extra)
    out = C.Output("bounds_references", args.tag)
    out(*C.header_lines(__file__, args.tag), "")
    record = dict(script=Path(__file__).name, cmgdb_version=C.cmgdb_version(), tag=args.tag, cases={})
    for name in args.cases:
        c = C.CASES[name]
        out(C.case_title(c))
        adaptive = run(c, c["model"], conley=False)
        rc = dict(fixed=[], uniform=[])
        for p in c["fixed"]:
            theory, lam, u, sign = C.theory_annotation(c, p)
            vs = [v for v, boxes in enumerate(adaptive["morse_sets"]) if any(C.contains(b, p) for b in boxes)]
            assert len(vs) == 1, (name, p, vs)
            chk = check_conditions(c, p, adaptive["morse_sets"][vs[0]])
            rc["fixed"].append(dict(point=list(p), eigenvalues=lam, unstable_dim=u, sign=sign,
                                    theory=theory, adaptive_morse_set=vs[0], conditions=chk))
            lams = ", ".join(f"{t:g}" for t in lam)
            out(f"  fixed point {C.fmt_point(p)}: eigenvalues of Df(p) {lams}; {u} expanding, "
                f"sign {sign:+d}; linearization {theory}")
            out(f"    Morse set {vs[0]} of the adaptive run, " +
                ("conditions (a) and (b) hold" if chk["applies"] else "conditions FAIL"))
            for line in describe(chk):
                out("      " + line)
        for d in c["uniform"]:
            r = run(c, (d, d, d))
            rows = []
            for p in c["fixed"]:
                vs = [v for v, boxes in enumerate(r["morse_sets"]) if any(C.contains(b, p) for b in boxes)]
                rows.append(dict(point=list(p), morse_sets=vs, annotations=[r["annotations"][v] for v in vs],
                                 conditions=[check_conditions(c, p, r["morse_sets"][v]) for v in vs]))
            rc["uniform"].append(dict(depth=d, run=r, fixed=rows))
            out(f"  uniform grid Model({d}, {d}, {d}): {r['n_cells']} cells of depths {r['depths']}, "
                f"{r['n_morse_sets']} Morse sets, {r['n_off_grid']} of {r['n_evaluations']} map "
                f"evaluations not boxes of the bisection tree")
            for row in rows:
                theory = C.theory_annotation(c, row["point"])[0]
                anns = "; ".join(f"Morse set {v}: {a}" for v, a in zip(row["morse_sets"], row["annotations"]))
                agree = row["annotations"] == [theory]
                cond = all(ch["applies"] for ch in row["conditions"])
                out(f"    fixed point {C.fmt_point(row['point'])}: {anns}"
                    f"{'  (= linearization)' if agree else '  <-- differs from the linearization'}"
                    f"{'' if cond else '  (conditions fail)'}")
            if c["dim"] == 1:
                for v, (boxes, a) in enumerate(zip(r["morse_sets"], r["annotations"])):
                    out(f"    Morse set {v}: {len(boxes)} cell(s), hull {C.fmt_box(C.hull(boxes))}, annotation {a}")
        record["cases"][name] = rc
        out("")
    out.json(record)
    out.close()


if __name__ == "__main__":
    main()
