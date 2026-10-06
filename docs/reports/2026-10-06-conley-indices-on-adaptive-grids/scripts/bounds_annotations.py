"""Conley indices of the fixed points in examples E1-E6, on the installed build.

For each example of bounds_common.CASES this script runs
ComputeConleyMorseGraph(Model(...)) with default options, as a user would,
and lists for every fixed point p the Morse sets whose boxes contain p and
their annotations, next to the annotation that the linearization at p
predicts.  If ComputeConleyMorseGraph raises (the fork release 1.3.3+fork.6
does on E2 and E5), the exception is recorded and the Morse sets are read
from ComputeMorseGraph, which computes the same Morse graph without the
Conley phase.  The box map is wrapped so that every rectangle it receives is
recorded, and the script counts the rectangles that are not boxes of the
bisection tree of the domain (the test of
tests/test_conley_geometry_regressions.py).  On the builds without the fix
these are the cubes of the Conley-phase complexes, placed with the wrong
bounds.

Usage (from any directory, with the Python of the build to test):

    python bounds_annotations.py [--tag TAG] [--boxmap] [--cases E1 E2 ...]

--boxmap uses CMGDB.BoxMap(f, rect, padding=False) as the box map instead
of the closed form of bounds_common.py.  Outputs:
outputs/bounds_annotations_<TAG>.txt and .json (with --boxmap,
bounds_annotations-boxmap_<TAG>.*).

Runs on upstream CMGDB 1.5.2, the fork release 1.3.3+fork.6, the merge
0e21503 and master 96642f2 of the fork (see bounds_common.py for the functions
it uses).  Runtime: about 1 s per build.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402


def run_case(c, boxmap):
    import CMGDB
    F = C.case_F(c, boxmap)
    rec = C.Recorder(F)
    error = None
    try:
        mg, mapg = CMGDB.ComputeConleyMorseGraph(C.make_model(c, rec))
    except Exception as e:  # recorded, not hidden
        error = f"{type(e).__name__}: {e}"
        mg, mapg = CMGDB.ComputeMorseGraph(C.make_model(c, F))
    lb, ub = c["lb"], c["ub"]
    off_grid = [r for r in rec.log if not C.is_subdivision_box(r, lb, ub)]
    grid = C.grid_boxes(mg, mapg)
    depth = {tuple(b): C.cell_index(b, lb, ub)[0] for b in grid}
    sets = []
    for v, boxes in enumerate(C.morse_sets(mg)):
        sets.append(dict(v=v, boxes=boxes, n_cells=len(boxes),
                         depths=sorted({depth[tuple(b)] for b in boxes}),
                         hull=C.hull(boxes),
                         annotation=None if error else [str(s) for s in mg.annotations(v)]))
    fixed = []
    for p in c["fixed"]:
        theory, lam, u, sign = C.theory_annotation(c, p)
        vs = [s["v"] for s in sets if any(C.contains(b, p) for b in s["boxes"])]
        fixed.append(dict(point=list(p), eigenvalues=lam, unstable_dim=u, theory=theory,
                          morse_sets=vs, annotations=[sets[v]["annotation"] for v in vs]))
    return dict(title=C.case_title(c), model=list(c["model"]), lb=lb, ub=ub, error=error,
                n_evaluations=len(rec.log), n_off_grid=len(off_grid), off_grid=off_grid,
                grid=sorted(grid), depths=sorted(set(depth.values())), morse_sets=sets,
                edges=C.morse_edges(mg), fixed=fixed)


def main():
    def extra(p):
        p.add_argument("--boxmap", action="store_true",
                       help="use CMGDB.BoxMap(f, rect, padding=False) as the box map")
        p.add_argument("--cases", nargs="*", default=list(C.CASES))
    args = C.parse_args(__doc__, extra)
    name = "bounds_annotations-boxmap" if args.boxmap else "bounds_annotations"
    out = C.Output(name, args.tag)
    out(*C.header_lines(__file__, args.tag))
    out("box map: " + ("CMGDB.BoxMap(f, rect, padding=False)" if args.boxmap
                       else "closed form of bounds_common.py"), "")
    record = dict(script=Path(__file__).name, cmgdb_version=C.cmgdb_version(), tag=args.tag,
                  boxmap=args.boxmap, cases={})
    for name_ in args.cases:
        c = C.CASES[name_]
        r = run_case(c, args.boxmap)
        record["cases"][name_] = r
        out(r["title"] + ("   (control)" if c.get("control") else ""))
        out(f"  final grid: {len(r['grid'])} cells of depths {r['depths']}; "
            f"{len(r['morse_sets'])} Morse sets; Morse graph edges {r['edges']}")
        out("  ComputeConleyMorseGraph: " + (f"raised {r['error']}" if r["error"] else "returned"))
        out(f"  box map evaluations: {r['n_evaluations']}, of which {r['n_off_grid']} "
            f"are not boxes of the bisection tree")
        if len(r["morse_sets"]) <= 12:
            for s in r["morse_sets"]:
                ann = s["annotation"] if s["annotation"] is not None else "(no annotation, exception)"
                out(f"  Morse set {s['v']}: {s['n_cells']} cell(s) of depth {s['depths']}, "
                    f"hull {C.fmt_box(s['hull'])}, annotation {ann}")
        for fp in r["fixed"]:
            lam = ", ".join(f"{t:g}" for t in fp["eigenvalues"])
            got = "; ".join(f"Morse set {v}: {a}" for v, a in zip(fp["morse_sets"], fp["annotations"]))
            agree = all(a == fp["theory"] for a in fp["annotations"]) and len(fp["annotations"]) == 1
            out(f"  fixed point {C.fmt_point(fp['point'])}: eigenvalues {lam}, "
                f"linearization {fp['theory']}; {got or 'in no Morse set'}"
                f"{'' if agree else '   <-- differs'}")
        out("")
    out.json(record)
    out.close()


if __name__ == "__main__":
    main()
