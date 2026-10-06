"""Outer approximation and monotonicity of the three box maps on the boxes that
the hierarchy evaluates.

For each box map (corner, padded, hull) and each of Model(12, 14, ...) and
Model(16, 18, ...), the script logs every call of the box map made by
ComputeMorseGraph (the hierarchy and the final map graph, without the Conley
index phase) and tests, for every distinct box B of depth d:

  outer      the hull of f(B) lies in Phi(B), so that f(B) subset Phi(B).
  monotone   for d >= 1, Phi(B) lies in Phi(B'), where B' is the box of depth
             d - 1 that contains B, the cell that the hierarchy bisected.

Each test measures how far the first box sticks out of the second, relative
to 1 + max |end point|, and counts the boxes where this excess is positive
and where it exceeds 1e-8.  The hull is the closed form of
leslie_common.hull_box without widening.  For the hull box map the outer test
is a tautology, so the script also checks the closed form against f on 3000
random boxes (f1 at a 60 x 60 grid of points and at 2000 points of each edge).

Output: outputs/leslie_monotonicity_<tag>.txt and .json.  Runs on every
build, in about 2 s on the Apple silicon machine of the stored outputs.
numpy only.

usage: python leslie_monotonicity.py --tag TAG
"""

import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
from leslie_common import LOWER, UPPER, THETA, BOX_MAPS, hull_box, fmt_box, histogram, depth_of


def lattice_key(b, q=2 ** 12):
    return (round((b[0] - LOWER[0]) / (UPPER[0] - LOWER[0]) * q), round((b[1] - LOWER[1]) / (UPPER[1] - LOWER[1]) * q),
            round((b[2] - LOWER[0]) / (UPPER[0] - LOWER[0]) * q), round((b[3] - LOWER[1]) / (UPPER[1] - LOWER[1]) * q))


TOL = 1e-8


def excess(inner, outer):
    """How far the box inner sticks out of the box outer, relative to
    1 + max |end point| (0 when inner lies in outer)."""
    e = max(outer[0] - inner[0], outer[1] - inner[1], inner[2] - outer[2], inner[3] - outer[3], 0.0)
    return e / (1.0 + max(abs(v) for v in list(inner) + list(outer)))


def check_hull_formula(out, rng, nboxes=3000):
    """Sampled values of f1 against the closed-form hull on random boxes."""
    a, b = THETA
    worst_out, worst_slack = -np.inf, 0.0
    for _ in range(nboxes):
        w = 10 ** rng.uniform(-3, 2)
        x0 = rng.uniform(LOWER[0], UPPER[0] - min(w, 89.0))
        y0 = rng.uniform(LOWER[1], UPPER[1] - min(w, 69.0))
        x1 = min(UPPER[0], x0 + w * rng.uniform(0.2, 1.0))
        y1 = min(UPPER[1], y0 + w * rng.uniform(0.2, 1.0))
        lo, _, hi, _ = hull_box([x0, y0, x1, y1], widen=0.0)
        X, Y = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 60))
        t = np.linspace(0.0, 1.0, 2000)
        ex = np.concatenate([x0 + (x1 - x0) * t, x0 + (x1 - x0) * t, np.full(2000, x0), np.full(2000, x1)])
        ey = np.concatenate([np.full(2000, y0), np.full(2000, y1), y0 + (y1 - y0) * t, y0 + (y1 - y0) * t])
        xs, ys = np.concatenate([X.ravel(), ex]), np.concatenate([Y.ravel(), ey])
        v = (a * xs + b * ys) * np.exp(-0.1 * (xs + ys))
        scale = 1.0 + max(abs(lo), abs(hi))
        worst_out = max(worst_out, (v.max() - hi) / scale, (lo - v.min()) / scale)
        worst_slack = max(worst_slack, (hi - v.max()) / scale)
    out(f"   closed-form hull against f1 on {nboxes} random boxes (widths 1e-3 to 1e2): largest excess of a sampled "
        f"value over the hull {worst_out:.1e} (relative); largest gap between the hull and the sampled maximum "
        f"{worst_slack:.1e} (relative)")
    return {"boxes": nboxes, "max_excess": float(worst_out), "max_gap": float(worst_slack)}


def main():
    import CMGDB
    args = LC.parse_args(__doc__)
    out = LC.Output("leslie_monotonicity", args.tag)
    out(*LC.header_lines(__file__, args.tag))
    rec = {"cmgdb_version": LC.cmgdb_version(), "tag": args.tag, "runs": []}
    for name in ("corner", "padded", "hull"):
        Phi = BOX_MAPS[name]
        for smin, smax in ((12, 14), (16, 18)):
            log = []

            def F(rect, Phi=Phi, log=log):
                log.append(list(rect))
                return Phi(rect)
            mg, gr = CMGDB.ComputeMorseGraph(LC.leslie_model(smin, smax, None, F))
            boxes = {}
            for r in log:
                boxes.setdefault(lattice_key(r), r)
            img = {k: Phi(list(r)) for k, r in boxes.items()}
            outer_miss, mono_miss, examples_outer, examples_mono = {}, {}, [], []
            missing_parent = 0
            for k, r in sorted(boxes.items()):
                d = depth_of(r)
                h = hull_box(r, widen=0.0)
                outer_miss[k] = excess(h, img[k])
                if outer_miss[k] > TOL:
                    examples_outer.append({"depth": d, "box": r, "Phi": img[k], "hull": h, "miss": outer_miss[k]})
                if d == 0:
                    continue
                pk = lattice_key(LC.parent_box(r))
                if pk not in img:
                    missing_parent += 1
                    img[pk] = Phi(LC.parent_box(r))
                mono_miss[k] = excess(img[k], img[pk])
                if mono_miss[k] > TOL:
                    examples_mono.append({"depth": d, "box": r, "Phi": img[k], "parent": LC.parent_box(r),
                                          "Phi_parent": img[pk], "miss": mono_miss[k]})
            depth = {k: depth_of(r) for k, r in boxes.items()}
            depths = histogram(depth.values())
            leaves = {lattice_key(mg.phase_space_box(i)) for i in range(gr.num_vertices())}

            def summary(miss):
                pos = [k for k, m in miss.items() if m > 0]
                big = [k for k, m in miss.items() if m > TOL]
                fine = [m for k, m in miss.items() if depth[k] >= 10]
                return {"positive": len(pos), "positive_by_depth": histogram(depth[k] for k in pos),
                        "above_tol": len(big), "above_tol_by_depth": histogram(depth[k] for k in big),
                        "above_tol_final_cells": len([k for k in big if k in leaves]),
                        "largest": max(miss.values()) if miss else 0.0,
                        "largest_depth_ge_10": max(fine) if fine else 0.0}
            so, sm = summary(outer_miss), summary(mono_miss)
            row = {"label": f"{name} {smin}/{smax}", "box_map": name, "min": smin, "max": smax, "calls": len(log),
                   "distinct_boxes": len(boxes), "boxes_by_depth": depths,
                   "deeper_than_min": sum(v for d, v in depths.items() if d > smin),
                   "final_cells": gr.num_vertices(), "outer": so, "monotone": sm,
                   "parents_not_evaluated": missing_parent, "not_outer": examples_outer, "not_monotone": examples_mono}
            rec["runs"].append(row)
            out("", f"{name} box map, Model({smin}, {smax}, lower, upper, F): {len(log)} calls, {len(boxes)} distinct "
                f"boxes ({row['deeper_than_min']} of depth > {smin}); final grid {gr.num_vertices()} cells; parents not "
                f"evaluated: {missing_parent}")
            out(f"   boxes by depth {depths}")
            for what, sv in (("not outer (f(B) not inside Phi(B))", so),
                             ("not monotone (Phi(B) not inside Phi(parent))", sm)):
                out(f"   {what}: by more than {TOL:g}: {sv['above_tol']} boxes, by depth {sv['above_tol_by_depth']}, "
                    f"{sv['above_tol_final_cells']} of them cells of the final grid")
                out(f"      by any positive amount: {sv['positive']} boxes, by depth {sv['positive_by_depth']}; "
                    f"largest excess {sv['largest']:.2e}, largest at depth >= 10: {sv['largest_depth_ge_10']:.2e}")
            if name == "corner" and smin == 12:
                for e in examples_outer:
                    out(f"      not outer, depth {e['depth']:2d}" + (" (final cell)" if lattice_key(e["box"]) in leaves else "")
                        + f": B = {fmt_box(e['box'])}, Phi(B) x-range [{e['Phi'][0]:.4f}, {e['Phi'][2]:.4f}], "
                        f"hull x-range [{e['hull'][0]:.4f}, {e['hull'][2]:.4f}]")
                for e in examples_mono:
                    out(f"      not monotone, depth {e['depth']:2d}: B = {fmt_box(e['box'])}, Phi(B) = {fmt_box(e['Phi'])}, "
                        f"Phi(parent) = {fmt_box(e['Phi_parent'])}")
    out("", "hull box map:")
    rec["hull_formula_check"] = check_hull_formula(out, np.random.default_rng(7))
    out.json(rec)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
