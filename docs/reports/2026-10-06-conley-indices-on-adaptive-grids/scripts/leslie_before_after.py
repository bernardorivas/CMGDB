"""Compare the outputs of the Leslie scripts across CMGDB builds.

Reads outputs/<name>_<tag>.json for name in leslie_morse, leslie_depth_one,
leslie_cycle, leslie_isolation, leslie_monotonicity and leslie_origin, and for
every tag lists the entries that differ from those of the first tag (the
reference, by default master 96642f2).  The version and the tag are not
compared, and grids and Morse sets are compared through the SHA-256 digests
of their boxes.  It ends with a table of the values that differ between the builds.

Output: outputs/leslie_before_after.txt and .json.  Needs no CMGDB.

usage: python leslie_before_after.py [--tags REF TAG ...]
"""

import json
import sys

sys.dont_write_bytecode = True

import leslie_common as LC

NAMES = ["leslie_morse", "leslie_depth_one", "leslie_cycle", "leslie_isolation", "leslie_monotonicity",
         "leslie_origin"]
DEFAULT_TAGS = ["cmgdb-1.5.3_fork.7.dev0-96642f2", "cmgdb-1.5.2", "cmgdb-1.3.3_fork.6",
                "cmgdb-1.5.3_fork.7.dev0-0e21503"]
IGNORE = {"cmgdb_version", "tag", "boxes", "cells"}     # boxes are compared through their digests


def short(v, n=150):
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


def diff(a, b, path):
    if type(a) is not type(b):
        yield path, a, b
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                yield f"{path}.{k}", a.get(k), b.get(k)
            else:
                yield from diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path, f"list of length {len(a)}", f"list of length {len(b)}"
            return
        for i, (x, y) in enumerate(zip(a, b)):
            key = x.get("label") if isinstance(x, dict) and "label" in x else i
            yield from diff(x, y, f"{path}[{key}]")
    elif a != b:
        yield path, a, b


def main():
    args = LC.parse_args(__doc__, lambda p: p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS))
    out = LC.Output("leslie_before_after")
    out(f"script: leslie_before_after.py", f"reference: {args.tags[0]}")
    data = {}
    for tag in args.tags:
        data[tag] = {}
        for name in NAMES:
            try:
                data[tag][name] = LC.load_json(name, tag)
            except FileNotFoundError:
                data[tag][name] = None
    versions = {tag: next((d["cmgdb_version"] for d in data[tag].values() if d), None) for tag in args.tags}
    for tag in args.tags:
        out(f"   {tag}: CMGDB {versions[tag]}, outputs found: {[n for n in NAMES if data[tag][n] is not None]}")
    ref = args.tags[0]
    summary = {}
    for tag in args.tags[1:]:
        out("", f"== {tag} against {ref}")
        summary[tag] = {}
        for name in NAMES:
            a, b = data[ref][name], data[tag][name]
            if a is None or b is None:
                out(f"   {name}: missing")
                continue
            diffs = list(diff(a, b, name))
            summary[tag][name] = [[p, x, y] for p, x, y in diffs]
            out(f"   {name}: {'identical' if not diffs else f'{len(diffs)} difference(s)'}")
            for p, x, y in diffs[:40]:
                out(f"      {p}:", f"         {ref}: {short(x)}", f"         {tag}: {short(y)}")
            if len(diffs) > 40:
                out(f"      ... and {len(diffs) - 40} more")

    out("", "Key values by build:")
    for tag in args.tags:
        m, c = data[tag]["leslie_morse"], data[tag]["leslie_cycle"]
        if not m or not c:
            continue
        runs = {r["label"]: r for r in m["runs"]}
        r12, r16 = runs["corner 12/14"], runs["corner 16/18"]
        out(f"   {tag}:")
        out(f"      corner 12/14, set around p*: annotation {r12['morse_sets'][c['M_vertex']]['annotation']}, "
            f"ComputeConleyIndexForCells: {short(c['index_for_cells_M'], 90)}")
        out(f"      corner 12/14, component of the final graph that contains it ({len(c['component_of_M']['cells'])} cells): "
            f"ComputeConleyIndexForCells {short(c['component_of_M']['index_for_cells'], 90)}")
        out(f"      corner 16/18 annotations: {[s['annotation'] for s in r16['morse_sets']]}")
        for label in ("hull 16/18", "padded 16/18"):
            if label in runs:
                out(f"      {label}: annotations {[s['annotation'] for s in runs[label]['morse_sets']]}, "
                    f"ComputeConleyIndexForCells {[short(s['index_for_cells'], 60) for s in runs[label]['morse_sets']]}")
    out.json({"tags": args.tags, "versions": versions, "differences": summary})
    out.close()


if __name__ == "__main__":
    sys.exit(main())
