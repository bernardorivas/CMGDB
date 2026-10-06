"""Tables across builds, from the outputs of the other scripts of Part I.

Reads outputs/bounds_annotations_<TAG>.json, bounds_annotations-boxmap_<TAG>.json,
bounds_references_<TAG>.json and bounds_mechanism_<TAG>.json for each build tag, and
writes outputs/bounds_table.txt and .json with

  1. the annotations of the Morse sets that contain the fixed points, on each
     build, next to the linearization and the uniform grids,
  2. checks across builds: whether the final grids, the Morse sets and the
     Morse graphs agree cell by cell, whether the annotations agree with those
     of bounds_mechanism.py (which sets CMGDB_MAPGRAPH_CACHE=0) and of the run
     with CMGDB.BoxMap, and whether the uniform grids yield the same
     annotations on every build,
  3. the number of map evaluations that are not boxes of the bisection tree,
  4. for each Morse set with a fixed point: the bounds of its complex before
     and after the fix, the index-pair condition for the maps on the cubes
     before and after the fix, and ComputeConleyIndex on each map, on each
     build, next to the annotation of the build.

Usage:  python bounds_table.py [--tags TAG ...] [--labels LABEL ...]
Needs no CMGDB.  Runtime: under 1 s.
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds_common as C  # noqa: E402

BUILDS = [("cmgdb-1.5.2", "upstream 1.5.2"),
          ("cmgdb-1.3.3_fork.6", "fork 1.3.3+fork.6"),
          ("cmgdb-1.5.3_fork.7.dev0-0e21503", "merge 0e21503"),
          ("cmgdb-1.5.3_fork.7.dev0-96642f2", "master 96642f2")]


def ann_text(a, error=None):
    if a is None:
        return error.split(":")[0] if error else "none"
    if a == []:
        return "[] (undefined)"
    return "[" + ", ".join(a) + "]"


def table(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)).rstrip()
    return [line(header), line(["-" * k for k in w])] + [line(r) for r in rows]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--tags", nargs="*", default=[t for t, _ in BUILDS])
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()
    labels = args.labels or [dict(BUILDS).get(t, t) for t in args.tags]
    out = C.Output("bounds_table")
    out(f"script: {Path(__file__).name}", "builds: " + "; ".join(f"{l} = {t}" for t, l in zip(args.tags, labels)), "")
    ann = {t: C.load_json("bounds_annotations", t) for t in args.tags}
    annb = {t: C.load_json("bounds_annotations-boxmap", t) for t in args.tags}
    ora = {t: C.load_json("bounds_references", t) for t in args.tags}
    mech = {t: C.load_json("bounds_mechanism", t) for t in args.tags}
    for t in args.tags:
        out(f"  {t}: CMGDB version {ann[t]['cmgdb_version']}")
    out("")
    record = dict(builds=dict(zip(args.tags, labels)), annotations=[], checks={}, off_grid={}, mechanism=[])

    # 1. annotations
    ref = ora[args.tags[-1]]
    header = ["example", "fixed point", "linearization", "uniform coarse", "uniform fine"] + labels
    rows = []
    for name in C.CASES:
        for k, p in enumerate(C.CASES[name]["fixed"]):
            theory = C.theory_annotation(C.CASES[name], p)[0]
            uni = []
            for u in ref["cases"][name]["uniform"]:
                a = u["fixed"][k]["annotations"]
                uni.append(ann_text(a[0]) if len(a) == 1 else str(a))
            row = [name, C.fmt_point(p), ann_text(theory)] + uni
            entry = dict(case=name, point=list(p), theory=theory,
                         uniform=[u["fixed"][k]["annotations"] for u in ref["cases"][name]["uniform"]], builds={})
            for t in args.tags:
                r = ann[t]["cases"][name]
                fp = r["fixed"][k]
                a = fp["annotations"][0] if len(fp["annotations"]) == 1 else fp["annotations"]
                text = ann_text(a, r["error"])
                row.append(text + ("" if a == theory else " *"))
                entry["builds"][t] = dict(annotation=a, error=r["error"])
            rows.append(row)
            record["annotations"].append(entry)
    out("1. Annotations of the Morse sets that contain the fixed points (* = differs from the linearization)", "")
    for line in table(rows, header):
        out("  " + line)
    out("", "  uniform coarse / fine: Model(d, d, d) with d = subdiv_min / subdiv_max of the example's model, "
        "from " + labels[-1] + " (the same on every build, see 2).", "")

    # 2. checks
    out("2. Checks across builds")
    chk = {}
    for name in C.CASES:
        base = ann[args.tags[0]]["cases"][name]
        same_grid = all(ann[t]["cases"][name]["grid"] == base["grid"] for t in args.tags)
        same_sets = all([s["boxes"] for s in ann[t]["cases"][name]["morse_sets"]] ==
                        [s["boxes"] for s in base["morse_sets"]] for t in args.tags)
        same_edges = all(ann[t]["cases"][name]["edges"] == base["edges"] for t in args.tags)
        same_mech = all([s["annotation"] for s in ann[t]["cases"][name]["morse_sets"]] ==
                        [s["annotation"] for s in mech[t]["cases"][name]["morse_sets"]] and
                        ann[t]["cases"][name]["grid"] == sorted(mech[t]["cases"][name]["grid"])
                        for t in args.tags)
        same_boxmap = all(annb[t]["cases"][name][k] == ann[t]["cases"][name][k]
                          for t in args.tags for k in ("grid", "morse_sets", "edges", "fixed", "error", "off_grid"))
        uni = lambda t: [[(f["morse_sets"], f["annotations"]) for f in u["fixed"]]
                         for u in ora[t]["cases"][name]["uniform"]]
        same_uniform = all(uni(t) == uni(args.tags[0]) for t in args.tags)
        uniform_all_depth = all(u["run"]["depths"] == [u["depth"]] and u["run"]["n_off_grid"] == 0
                                for t in args.tags for u in ora[t]["cases"][name]["uniform"])
        cond = all(f["conditions"]["applies"] for f in ora[args.tags[-1]]["cases"][name]["fixed"])
        chk[name] = dict(grid=same_grid, morse_sets=same_sets, edges=same_edges, mechanism_run=same_mech,
                         boxmap=same_boxmap, uniform=same_uniform, uniform_single_depth_on_grid=uniform_all_depth,
                         linearization_conditions=cond)
        out(f"  {name}: final grid {len(base['grid'])} cells, identical on all builds: {same_grid}; "
            f"Morse sets identical: {same_sets}; Morse graph edges identical: {same_edges}")
        out(f"      annotations and grid of bounds_mechanism.py (CMGDB_MAPGRAPH_CACHE=0) identical: {same_mech}; "
            f"run with CMGDB.BoxMap identical in every field: {same_boxmap}")
        out(f"      uniform grids: same Morse sets and annotations at the fixed points on all builds: {same_uniform}; "
            f"one depth and no evaluation off the tree: {uniform_all_depth}; "
            f"conditions (a) and (b) of bounds_references.py hold at every fixed point: {cond}")
    record["checks"] = chk
    out("")

    # 3. off-grid evaluations
    out("3. Map evaluations during ComputeConleyMorseGraph that are not boxes of the bisection tree "
        "(off-grid / total; default options)", "")
    rows = []
    for name in C.CASES:
        row = [name]
        for t in args.tags:
            r = ann[t]["cases"][name]
            row.append(f"{r['n_off_grid']} / {r['n_evaluations']}" + (" (raised)" if r["error"] else ""))
            record["off_grid"].setdefault(name, {})[t] = [r["n_off_grid"], r["n_evaluations"]]
        rows.append(row)
    for line in table(rows, ["example"] + labels):
        out("  " + line)
    out("")

    # 4. mechanism
    out("4. The complexes of the Morse sets with a fixed point (reconstructed; identical on every build, see the "
        "consistency column)", "")
    rows = []
    for name in C.CASES:
        sets = mech[args.tags[-1]]["cases"][name]["morse_sets"]
        for v, ms in enumerate(sets):
            if not ms["fixed"]:
                continue
            an = ms["analysis"]
            D = len(an["n"])
            consistent = all(json.dumps(mech[t]["cases"][name]["morse_sets"][v]["analysis"]["maps"]) ==
                             json.dumps(an["maps"]) and
                             mech[t]["cases"][name]["morse_sets"][v]["analysis"]["bounds_before"] == an["bounds_before"]
                             for t in args.tags)
            ob, tb = an["bounds_before"], an["bounds_after"]
            short = [d for d in range(D) if ob[d] != tb[d] or ob[D + d] != tb[D + d]]
            bounds = "; ".join(f"axis {d}: [{ob[d]:.6g}, {ob[D + d]:.6g}] vs [{tb[d]:.6g}, {tb[D + d]:.6g}]"
                               for d in short) or "exact"
            ip_b = an["checks"]["before"]["index_pair"]
            ip_a = an["checks"]["after"]["index_pair"]
            viol = "; ".join(f"{tuple(a)}->{[tuple(s) for s in ss]}" for a, ss in an["checks"]["before"]["exit_cubes_into_S"])
            empty = [tuple(q) for q in an["checks"]["before"]["empty"]]
            row = [name, C.fmt_point(ms["fixed"][0]), f"{len(an['deeper'])}/{len(an['X'])}", str(an["n"]), bounds,
                   ("yes" if ip_b else f"no: {viol}") + (f"; empty {empty}" if empty else ""),
                   "yes" if ip_a else "no", consistent]
            rows.append(row)
            entry = dict(case=name, v=v, point=ms["fixed"][0], deeper=len(an["deeper"]), X=len(an["X"]), n=an["n"],
                         bounds_before=ob, bounds_after=tb, index_pair_before=ip_b, index_pair_after=ip_a,
                         exit_cubes_into_S=an["checks"]["before"]["exit_cubes_into_S"], empty_before=empty,
                         reconstruction_identical_on_builds=consistent, builds={})
            for t in args.tags:
                m2 = mech[t]["cases"][name]["morse_sets"][v]
                rec = m2["recorded"]
                entry["builds"][t] = dict(annotation=m2["annotation"],
                                          conley_index_before=m2["analysis"]["conley_index"]["before"],
                                          conley_index_after=m2["analysis"]["conley_index"]["after"],
                                          recorded_bounds=rec["bounds_used"] if rec else None,
                                          recorded_map=rec["map_used"] if rec else None,
                                          max_deviation=rec["max_deviation"] if rec else None,
                                          cube_evaluations=[rec["n_cube_evaluations"], rec["n_distinct"]] if rec else None)
            record["mechanism"].append(entry)
    for line in table(rows, ["example", "fixed point", "deeper/|X|", "n", "bounds before vs after the fix (short axes)",
                             "index pair, map before the fix", "after", "same on all builds"]):
        out("  " + line)
    out("")
    out("  ComputeConleyIndex on the map on the cubes before / after the fix, and the annotation of "
        "ComputeConleyMorseGraph, per build:", "")
    rows = []
    for e in record["mechanism"]:
        row = [e["case"], C.fmt_point(e["point"])]
        for t in args.tags:
            b = e["builds"][t]
            err = ann[t]["cases"][e["case"]]["error"]
            row.append(f"{ann_text(b['conley_index_before']) if isinstance(b['conley_index_before'], list) else b['conley_index_before'].split(':')[0]}"
                       f" / {ann_text(b['conley_index_after'])} | {ann_text(b['annotation'], err)}")
        rows.append(row)
    for line in table(rows, ["example", "fixed point"] + labels):
        out("  " + line)
    out("")
    out("  Boxes the map received for the cubes (recorded), per build: bounds they come from, the map they yield, and "
        "the number of evaluations on cubes (all and distinct rectangles):", "")
    rows = []
    for e in record["mechanism"]:
        row = [e["case"], C.fmt_point(e["point"])]
        for t in args.tags:
            b = e["builds"][t]
            if b["recorded_bounds"] is None:
                row.append("not reached")
            else:
                ev = b["cube_evaluations"]
                row.append(f"{'/'.join(b['recorded_bounds'])}; map {b['recorded_map']}; {ev[0]} evals, {ev[1]} distinct")
        rows.append(row)
    for line in table(rows, ["example", "fixed point"] + labels):
        out("  " + line)
    devs = [m["recorded"]["max_deviation"] for t in args.tags for name in C.CASES
            for m in mech[t]["cases"][name]["morse_sets"] if m["recorded"]]
    unmatched = sum(len(m["recorded"]["unmatched"]) for t in args.tags for name in C.CASES
                    for m in mech[t]["cases"][name]["morse_sets"] if m["recorded"])
    out("", f"  Over all builds and Morse sets: {len(devs)} Morse sets with recorded cube evaluations, "
        f"{unmatched} recorded rectangles that match no predicted rectangle, largest deviation from the "
        f"predicted rectangle {max(devs):.1e}.")
    record["recorded_summary"] = dict(n_sets=len(devs), unmatched=unmatched, max_deviation=max(devs))
    out.json(record)
    out.close()


if __name__ == "__main__":
    main()
