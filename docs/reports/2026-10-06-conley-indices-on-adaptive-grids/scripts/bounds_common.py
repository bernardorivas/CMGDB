"""Definitions shared by the scripts of Part I, the bounds of the cubical complex.

The commit "Bound the relative complex by the cubes it holds" (577d988) of
github.com/bernardorivas/CMGDB fixes the defect of Part I.  CMGDB computes
the Conley index of a Morse set S from the pair X = cover(F(S)), A = X \\ S
of cells of the final grid.  TreeGrid::relativeComplex turns the pair into a
cubical complex at the depth d of S: a cell deeper than d enters as its
ancestor of depth d, a shallower cell as its descendants of depth d.
CubicalComplex::geometryOfCube assigns to the cube with offset k along an
axis the interval

    lo + [k, k + 1] (hi - lo) / n,

where n is the number of cubes along the axis and [lo, hi] are the bounds
of the complex.  Before the fix the bounds were the hull of the cells of
X, which misses part of the ancestor of a cell deeper than d.

Examples.  f(x) = x / (2 - x), in each coordinate in dimension 2, has the
fixed points 0 (f'(0) = 1/2) and 1 (f'(1) = 2).  The logistic map
g(x) = 3.2 x (1 - x) has the fixed point p = 11/16 with g'(p) = -1.2.

  E1  f on [0, 1.25],        Model(4, 8, 0, 10000)
  E2  f on [0, 1.2],         Model(4, 8, 0, 10000)
  E3  g on [0.1, 0.85],      Model(4, 8, 0, 10000)
  E4  f on [0, 1.25]^2,      Model(8, 14, 4, 10000)
  E5  f on [0, 1.2]^2,       Model(8, 12, 4, 10000)
  E6  f on [0, 1.2]^2,       Model(6, 10, 4, 10000), the model of
      examples/Lattice_and_Nontrivial_CMGraph.ipynb (control)

Box maps.  For f, F(B) is the box spanned by the images of the lower and the
upper corner of B.  Since f is increasing in each coordinate, this is
CMGDB.BoxMap(f, B, padding=False) value for value, and it does not depend on
the build.  For g, F(B) is the interval spanned by g at the end points of B
and at 1/2 when 1/2 lies in B, which is g(B) up to rounding.  The option
--boxmap of bounds_annotations.py uses CMGDB.BoxMap(., B, padding=False) instead.

Builds.  The scripts run on each of these builds (default tags in brackets):

    upstream CMGDB 1.5.2     [cmgdb-1.5.2]         pip install CMGDB==1.5.2
    fork release 1.3.3+fork.6 [cmgdb-1.3.3_fork.6]  pip install cmgdb==1.3.3+fork.6
        --find-links https://github.com/bernardorivas/CMGDB/releases/expanded_assets/v1.3.3%2Bfork.6
    fork merge 0e21503 and fork master 96642f2 (both report the version
        1.5.3+fork.7.dev0, so pass --tag cmgdb-1.5.3_fork.7.dev0-<commit>)
        pip install git+https://github.com/bernardorivas/CMGDB@<commit>

They use only what these builds have in common: Model, ComputeMorseGraph,
ComputeConleyMorseGraph, ComputeConleyIndex, BoxMap, the MorseGraph methods
num_vertices, morse_set_boxes, annotations, adjacencies and
phase_space_box, and MapGraph.num_vertices.

Outputs go to ../outputs and figures to ../figures.  Every output name ends
with a build tag, given by --tag (default "cmgdb-" + the installed CMGDB
version with "+" replaced by "_"), and every text output starts with the
CMGDB version it was computed with.
"""

import argparse
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
FIGURES = HERE.parent / "figures"
P_FIELD = 5          # CMGDB computes homology with coefficients in F_5
LIMIT = 10000

# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

R_LOGISTIC = 3.2


def f_xs(t):
    return t / (2.0 - t)


def df_xs(t):
    return 2.0 / (2.0 - t) ** 2


def F_xs(rect):
    """Box spanned by the images of the lower and upper corner of rect."""
    D = len(rect) // 2
    return [f_xs(rect[d]) for d in range(D)] + [f_xs(rect[D + d]) for d in range(D)]


def f_log(t):
    return R_LOGISTIC * t * (1.0 - t)


def df_log(t):
    return R_LOGISTIC * (1.0 - 2.0 * t)


def F_log(rect):
    """Interval spanned by g at the end points of rect and at its critical
    point 1/2 when 1/2 lies in rect."""
    a, b = rect[0], rect[1]
    pts = [a, b] + ([0.5] if a < 0.5 < b else [])
    vals = [f_log(t) for t in pts]
    return [min(vals), max(vals)]


def boxmap_F(name):
    """CMGDB.BoxMap(f, rect, padding=False) for the map called name."""
    import CMGDB
    if name == "xs":
        fv = lambda x: [f_xs(t) for t in x]
    else:
        fv = lambda x: [f_log(t) for t in x]
    return lambda rect: [float(t) for t in CMGDB.BoxMap(fv, list(rect), padding=False)]


MAPS = {"xs": dict(F=F_xs, f=f_xs, df=df_xs, text="x/(2-x)"),
        "logistic": dict(F=F_log, f=f_log, df=df_log, text="3.2 x (1-x)")}

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

_P_LOG = 1.0 - 1.0 / R_LOGISTIC        # 11/16 = 0.6875
_CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

CASES = {
    "E1": dict(map="xs", lb=[0.0], ub=[1.25], model=(4, 8, 0), uniform=(4, 8),
               fixed=[(0.0,), (1.0,)]),
    "E2": dict(map="xs", lb=[0.0], ub=[1.2], model=(4, 8, 0), uniform=(4, 8),
               fixed=[(0.0,), (1.0,)]),
    "E3": dict(map="logistic", lb=[0.1], ub=[0.85], model=(4, 8, 0), uniform=(4, 8),
               fixed=[(_P_LOG,)]),
    "E4": dict(map="xs", lb=[0.0, 0.0], ub=[1.25, 1.25], model=(8, 14, 4), uniform=(8, 14),
               fixed=_CORNERS),
    "E5": dict(map="xs", lb=[0.0, 0.0], ub=[1.2, 1.2], model=(8, 12, 4), uniform=(8, 12),
               fixed=_CORNERS),
    "E6": dict(map="xs", lb=[0.0, 0.0], ub=[1.2, 1.2], model=(6, 10, 4), uniform=(6, 10),
               fixed=_CORNERS, control=True),
}
for _name, _c in CASES.items():
    _c["name"] = _name
    _c["dim"] = len(_c["lb"])


def case_title(c):
    m = MAPS[c["map"]]["text"]
    dom = " x ".join(f"[{a:g}, {b:g}]" for a, b in zip(c["lb"], c["ub"]))
    smin, smax, sinit = c["model"]
    return f"{c['name']}: f(x) = {m} on {dom}, Model({smin}, {smax}, {sinit}, {LIMIT})"


def case_F(c, boxmap=False):
    return boxmap_F(c["map"]) if boxmap else MAPS[c["map"]]["F"]


def make_model(c, F, subdiv=None):
    import CMGDB
    smin, smax, sinit = subdiv if subdiv is not None else c["model"]
    return CMGDB.Model(smin, smax, sinit, LIMIT, list(c["lb"]), list(c["ub"]), F)


def fmt_point(p):
    return "(" + ", ".join(f"{t:g}" for t in p) + ")"


# ---------------------------------------------------------------------------
# Linearization at the fixed points
# ---------------------------------------------------------------------------

def theory_annotation(c, p):
    """The annotation that the linearization at the hyperbolic fixed point p
    predicts.  f is a product of maps of the line, so the eigenvalues of
    Df(p) are the derivatives f'(p_i).  With u expanding coordinates the
    Conley index of {p} is F_5 in degree u, and the map on it is the sign
    of the product of the expanding eigenvalues.  CMGDB writes the
    invariant factors of that map, with the factors x removed, and "0"
    when none is left, so the annotation is "x-1" (sign +1) or "x+1"
    (sign -1) in degree u and "0" in every other degree."""
    df = MAPS[c["map"]]["df"]
    lam = [df(t) for t in p]
    assert all(abs(abs(v) - 1.0) > 1e-9 for v in lam), "p is not hyperbolic"
    unstable = [v for v in lam if abs(v) > 1.0]
    sign = 1
    for v in unstable:
        sign *= 1 if v > 0 else -1
    u = len(unstable)
    ann = ["0"] * (len(p) + 1)
    ann[u] = "x-1" if sign > 0 else "x+1"
    return ann, lam, u, sign


# ---------------------------------------------------------------------------
# Cells of the bisection tree
# ---------------------------------------------------------------------------

def splits_at_depth(k, D):
    """Bisections per axis of a cell of depth k.  The tree splits the axes in
    turn, starting with the first."""
    return [k // D + (1 if d < k % D else 0) for d in range(D)]


def cell_index(box, lb, ub):
    """(depth, offsets) of a box of the bisection tree of [lb, ub]: the box
    is the product over d of [lb + j h, lb + (j + 1) h] with
    h = (ub - lb) / 2^s_d and s_d = splits_at_depth(depth)[d]."""
    D = len(lb)
    sp, off = [], []
    for d in range(D):
        ratio = (ub[d] - lb[d]) / (box[D + d] - box[d])
        s = int(round(math.log2(ratio)))
        if not math.isclose(ratio, 2.0 ** s, rel_tol=1e-9):
            raise ValueError(f"not a box of the bisection tree: {box}")
        h = (ub[d] - lb[d]) / 2 ** s
        x = (box[d] - lb[d]) / h
        j = int(round(x))
        if not math.isclose(x, j, abs_tol=1e-6):
            raise ValueError(f"not a box of the bisection tree: {box}")
        sp.append(s)
        off.append(j)
    k = sum(sp)
    if sp != splits_at_depth(k, D):
        raise ValueError(f"not a box of the bisection tree: {box}")
    return k, off


def is_subdivision_box(rect, lb, ub):
    """True if rect is a node of the bisection tree of the domain
    (the test of tests/test_conley_geometry_regressions.py)."""
    try:
        cell_index(rect, lb, ub)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def tree_node_geometry(depth, off, lb, ub):
    """The box of the tree node of the given depth and offsets, computed as
    TreeGrid::geometryOfTreeNode does: as convex combinations of the bounds
    with coefficients j / 2^s."""
    D = len(lb)
    sp = splits_at_depth(depth, D)
    lo, hi = [], []
    for d in range(D):
        a = off[d] / 2 ** sp[d]                 # fraction below the box
        b = 1.0 - (off[d] + 1) / 2 ** sp[d]     # fraction above the box
        lo.append(a * ub[d] + (1.0 - a) * lb[d])
        hi.append(b * lb[d] + (1.0 - b) * ub[d])
    return lo + hi


# ---------------------------------------------------------------------------
# TreeGrid::cover (cells of the grid that meet a rectangle)
# ---------------------------------------------------------------------------

_WIDTH = 2 ** 60          # INTPHASEWIDTH in TreeGrid.h
_SLACK = 2 ** 10          # TRUNCATIONERROR in TreeGrid.h


def tree_cover(rect, grid, lb, ub):
    """Indices of the cells of grid (a list of (depth, offsets)) that
    TreeGrid::cover returns for rect, without periodicity.

    TreeGrid::coverAccept(RectGeo) maps rect to [0, 1]^D, returns nothing if
    rect lies beyond a face of the domain, clamps it to [0, 1]^D, scales it
    to integer coordinates on [0, 2^60] and widens it by 2^10 on each side.
    A cell is returned when its closed integer box meets the widened
    rectangle in every direction.  So the cover is the set of cells that
    meet the closed rectangle widened by 2^-50 of the domain width."""
    D = len(lb)
    LB, UB = [], []
    for d in range(D):
        w = ub[d] - lb[d]
        lo = (rect[d] - lb[d]) / w
        hi = (rect[D + d] - lb[d]) / w
        if hi < 0.0 or lo > 1.0:
            return []
        lo = min(max(lo, 0.0), 1.0)
        hi = min(max(hi, 0.0), 1.0)
        L = int(float(_WIDTH) * lo) - _SLACK
        U = int(float(_WIDTH) * hi) + _SLACK
        LB.append(max(L, 0))
        UB.append(min(U, _WIDTH))
    out = []
    for i, (k, off) in enumerate(grid):
        sp = splits_at_depth(k, D)
        ok = True
        for d in range(D):
            size = _WIDTH >> sp[d]
            if LB[d] > (off[d] + 1) * size or UB[d] < off[d] * size:
                ok = False
                break
        if ok:
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# The cubical complex of the Conley index computation
# ---------------------------------------------------------------------------

def cubes_of_cell(k, off, depth, D):
    """Cubes of depth `depth` that TreeGrid::GridElementToCubes produces for a
    cell of depth k: its ancestor if k >= depth, else its descendants.
    Cubes are tuples of offsets along the axes."""
    sk, sd = splits_at_depth(k, D), splits_at_depth(depth, D)
    if k >= depth:
        return [tuple(off[d] >> (sk[d] - sd[d]) for d in range(D))]
    ranges = [range(off[d] << (sd[d] - sk[d]), (off[d] + 1) << (sd[d] - sk[d])) for d in range(D)]
    return [tuple(q) for q in itertools.product(*ranges)]


def cube_geometry(q, bounds, n):
    """CubicalComplex::geometryOfCube: the cube with offsets q in a complex
    with the given bounds and n[d] cubes along axis d."""
    D = len(n)
    lo, hi = [], []
    for d in range(D):
        scale = (bounds[D + d] - bounds[d]) / float(n[d])
        a = bounds[d] + scale * q[d]
        lo.append(a)
        hi.append(a + scale)
    return lo + hi


def complex_cover(image, bounds, n, present):
    """CubicalComplex::cover (non-periodic): the cubes of the complex that
    CMGDB assigns to an image.

    On each axis the end points of the image are clamped to [lo, hi], and
    the offsets from floor(n (a - lo) / (hi - lo)) to floor(n (b - lo) /
    (hi - lo)) are kept, computed in double precision as in the C++
    (scale = n / (hi - lo), then (uint64_t)(scale * (a - lo))).  Offset n
    is the padding layer and holds no cube.  So a cube is assigned when the
    clamped image meets the half-open box prod_d [lo + k h, lo + (k+1) h),
    h = (hi - lo) / n: a cube that the image meets only on its upper face
    is not assigned, an image that lies wholly above hi gets no cube, and
    one that lies wholly below lo gets the first layer."""
    D = len(n)
    rng = []
    for d in range(D):
        lowbound, highbound = bounds[d], bounds[D + d]
        scale = float(n[d]) / (highbound - lowbound)
        lower = image[d] - lowbound
        if lower < 0:
            lower = 0.0
        if lower + lowbound > highbound:
            lower = highbound - lowbound
        upper = image[D + d] - lowbound
        if upper < 0:
            upper = 0.0
        if upper + lowbound > highbound:
            upper = highbound - lowbound
        rng.append(range(int(scale * lower), int(scale * upper) + 1))
    return sorted(tuple(q) for q in itertools.product(*rng) if tuple(q) in present)


def closed_cover(image, bounds, n, present):
    """Cubes whose closed box meets the image clamped to the bounds.  CMGDB
    does not use this rule.  It serves to show where the result depends on
    an image that touches a face of a cube."""
    D = len(n)
    r = [min(max(image[d], bounds[d]), bounds[D + d]) for d in range(D)] + \
        [min(max(image[D + d], bounds[d]), bounds[D + d]) for d in range(D)]
    out = []
    for q in present:
        g = cube_geometry(q, bounds, n)
        if all(g[d] <= r[D + d] and r[d] <= g[D + d] for d in range(D)):
            out.append(tuple(q))
    return sorted(out)


def flat_index(q, n):
    """Index of cube q in ComputeConleyIndex: q[0] + n[0] (q[1] + n[1] (...))."""
    idx, mul = 0, 1
    for d in range(len(n)):
        idx += q[d] * mul
        mul *= n[d]
    return idx


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class Recorder:
    """A box map that keeps a list of every rectangle it receives."""

    def __init__(self, F):
        self.F = F
        self.log = []

    def __call__(self, rect):
        r = [float(t) for t in rect]
        self.log.append(r)
        return self.F(r)


def grid_boxes(mg, mapg):
    return [[float(t) for t in mg.phase_space_box(i)] for i in range(mapg.num_vertices())]


def morse_sets(mg):
    return [sorted([float(t) for t in b] for b in mg.morse_set_boxes(v))
            for v in range(mg.num_vertices())]


def morse_edges(mg):
    return sorted((v, int(w)) for v in range(mg.num_vertices()) for w in mg.adjacencies(v))


def contains(box, p):
    D = len(p)
    return all(box[d] <= p[d] <= box[D + d] for d in range(D))


def hull(boxes):
    D = len(boxes[0]) // 2
    return [min(b[d] for b in boxes) for d in range(D)] + [max(b[D + d] for b in boxes) for d in range(D)]


# ---------------------------------------------------------------------------
# Command line, build tag and output files
# ---------------------------------------------------------------------------

def cmgdb_version():
    try:
        return importlib.metadata.version("CMGDB")
    except importlib.metadata.PackageNotFoundError:
        return None


def default_tag():
    v = cmgdb_version()
    return "cmgdb-" + v.replace("+", "_") if v else "no-cmgdb"


def parse_args(doc, extra=None):
    parser = argparse.ArgumentParser(description=doc.strip().splitlines()[0] if doc else None)
    parser.add_argument("--tag", default=None,
                        help="build tag appended to output names (default: cmgdb-<version>)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    if args.tag is None:
        args.tag = default_tag()
    return args


def header_lines(script, tag, cmgdb=True, env=True):
    """The header of an output.  With env, it also lists the CMGDB_MAPGRAPH_*
    variables set in the environment, which change the number of box map
    evaluations.  The stored outputs were computed with none of them set."""
    lines = [f"script: {Path(script).name}"]
    if cmgdb:
        lines.append(f"CMGDB version: {cmgdb_version()}   build tag: {tag}")
    lines.append(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    mapgraph = sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith("CMGDB_MAPGRAPH_"))
    if cmgdb and env and mapgraph:
        lines.append("environment: " + ", ".join(mapgraph))
    return lines


class Output:
    """Writes lines to stdout and to outputs/<name>_<tag>.txt, and a JSON
    record to outputs/<name>_<tag>.json."""

    def __init__(self, name, tag=None):
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_{tag}" if tag else name
        self.txt_path = OUTPUTS / f"{stem}.txt"
        self.json_path = OUTPUTS / f"{stem}.json"
        self._fh = open(self.txt_path, "w")

    def __call__(self, *lines):
        for line in lines:
            print(line)
            self._fh.write(str(line) + "\n")
        self._fh.flush()

    def json(self, obj):
        with open(self.json_path, "w") as fh:
            json.dump(obj, fh, indent=None, separators=(",", ":"))
            fh.write("\n")

    def close(self):
        self._fh.close()


def load_json(name, tag=None):
    stem = f"{name}_{tag}" if tag else name
    with open(OUTPUTS / f"{stem}.json") as fh:
        return json.load(fh)


def fmt(x, nd=6):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(y, nd) for y in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def fmt_box(b, nd=6):
    D = len(b) // 2
    return " x ".join(f"[{b[d]:.{nd}f}, {b[D + d]:.{nd}f}]" for d in range(D))
