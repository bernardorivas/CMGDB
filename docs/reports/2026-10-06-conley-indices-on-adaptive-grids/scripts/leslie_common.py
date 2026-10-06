"""Definitions shared by the Leslie-map scripts (Part II).

The map is

    f(x, y) = ((19.6 x + 23.68 y) exp(-0.1 (x + y)), 0.7 x)

on Q = [-0.001, 90] x [-0.001, 70], with the model of `leslie_model` in
tests/test_map_graph_cache.py: Model(subdiv_min, subdiv_max, lower, upper, F)
with F(rect) = CMGDB.BoxMap(f, rect), so the initial subdivision is 0 and the
size limit is 10000.  A box is a list [x0, y0, x1, y1].

Box maps (keys of BOX_MAPS):

  corner  CMGDB.BoxMap(f, B): the smallest box that contains f at the four
          corners of B.
  padded  CMGDB.BoxMap(f, B, padding=True): the corner box widened on each
          side by the widths of B (by x1 - x0 in the first coordinate and by
          y1 - y0 in the second).
  hull    the smallest box that contains f(B), from the closed form below,
          widened by 1e-9 (1 + max |end point|) in the first coordinate and
          by 1e-12 in the second.

Closed form of the hull.  Write f1(x, y) = (a x + b y) exp(-0.1 (x + y)) with
a = 19.6 and b = 23.68.  Then

    d f1/dx = (a - 0.1 s) e,   d f1/dy = (b - 0.1 s) e,   s = a x + b y,

so a critical point would need a = b, and f1 has none.  Its extreme values on
a box lie on the edges.  On an edge y = c the derivative along the edge
vanishes only at x = 10 - (b/a) c, where f1 restricted to the edge has its
maximum, and on an edge x = c only at y = 10 - (a/b) c.  So the minimum of f1
on B is attained at a corner, and its maximum at a corner or at one of these
at most four edge points.  The second coordinate 0.7 x ranges over
[0.7 x0, 0.7 x1].  The widening covers the rounding error of exp.

The scripts write their outputs to ../outputs and their figures to
../figures.  Every output file name ends with a build tag, given by --tag
(default: "cmgdb-" + the installed CMGDB version with "+" replaced by "_"),
and every text output starts with the CMGDB version it was computed with.
Master 96642f2 and the merge 0e21503 both report the version
1.5.3+fork.7.dev0, so their runs pass the tags
cmgdb-1.5.3_fork.7.dev0-96642f2 and cmgdb-1.5.3_fork.7.dev0-0e21503.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
FIGURES = HERE.parent / "figures"

THETA = (19.6, 23.68)
LOWER = (-0.001, -0.001)
UPPER = (90.0, 70.0)
LIMIT = 10000                      # default size limit of Model


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

def f(x):
    """The Leslie map, written as in tests/test_map_graph_cache.py."""
    s = x[0] + x[1]
    return [(THETA[0] * x[0] + THETA[1] * x[1]) * math.exp(-0.1 * s),
            0.7 * x[0]]


def f1(x, y):
    return (THETA[0] * x + THETA[1] * y) * math.exp(-0.1 * (x + y))


def Df(x, y):
    """Jacobian matrix of f at (x, y), as nested lists."""
    a, b = THETA
    e = math.exp(-0.1 * (x + y))
    s = a * x + b * y
    return [[(a - 0.1 * s) * e, (b - 0.1 * s) * e], [0.7, 0.0]]


def f_array(P):
    """f on an (..., 2) numpy array of points."""
    import numpy as np
    x, y = P[..., 0], P[..., 1]
    return np.stack([(THETA[0] * x + THETA[1] * y) * np.exp(-0.1 * (x + y)),
                     0.7 * x], axis=-1)


def matmul2(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def eig2(M):
    """Eigenvalues (complex) of a 2 x 2 matrix, trace and determinant."""
    import cmath
    t = M[0][0] + M[1][1]
    d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    r = cmath.sqrt(t * t - 4 * d)
    return (t + r) / 2, (t - r) / 2, t, d


def iterate_with_jacobian(z, n):
    """f^n(z) and D(f^n)(z)."""
    x, y = z
    J = [[1.0, 0.0], [0.0, 1.0]]
    for _ in range(n):
        J = matmul2(Df(x, y), J)
        x, y = f((x, y))
    return (x, y), J


def periodic_orbit(seed, n, steps=60):
    """Newton's method for f^n(z) = z from seed.  Returns the orbit
    [z, f(z), ..., f^(n-1)(z)] and the residual |f^n(z) - z|."""
    z = list(seed)
    for _ in range(steps):
        (xn, yn), J = iterate_with_jacobian(z, n)
        g0, g1 = xn - z[0], yn - z[1]
        m00, m01, m10, m11 = J[0][0] - 1.0, J[0][1], J[1][0], J[1][1] - 1.0
        det = m00 * m11 - m01 * m10
        z = [z[0] - (m11 * g0 - m01 * g1) / det, z[1] - (-m10 * g0 + m00 * g1) / det]
    (xn, yn), _ = iterate_with_jacobian(z, n)
    orbit = [tuple(z)]
    for _ in range(n - 1):
        orbit.append(tuple(f(orbit[-1])))
    return orbit, math.hypot(xn - z[0], yn - z[1])


# Fixed points: f(z) = z if and only if y = 0.7 x and x = 36.176 x exp(-0.17 x).
X_STAR = math.log(THETA[0] + 0.7 * THETA[1]) / 0.17
P_STAR = (X_STAR, 0.7 * X_STAR)
ORIGIN = (0.0, 0.0)
# The attracting and the saddle 3-cycle, refined by Newton's method.
A3, A3_RESIDUAL = periodic_orbit((7.5, 0.5), 3)
S3, S3_RESIDUAL = periodic_orbit((19.2, 1.3), 3)
NAMED_POINTS = {"0": [ORIGIN], "p*": [P_STAR], "a3": A3, "s3": S3}


def gamma_orbit(n, step=1, burn=100000, seed=(30.0, 20.0)):
    """n points of the orbit of seed after burn steps, every step-th point.
    The orbit accumulates on the attracting invariant circle around p*."""
    x, y = seed
    for _ in range(burn):
        x, y = f((x, y))
    out = []
    for k in range(n * step):
        if k % step == 0:
            out.append((x, y))
        x, y = f((x, y))
    return out


# ---------------------------------------------------------------------------
# Box maps
# ---------------------------------------------------------------------------

def _cmgdb():
    import CMGDB
    return CMGDB


def corner_box_map(rect):
    return _cmgdb().BoxMap(f, rect)


def padded_box_map(rect):
    return _cmgdb().BoxMap(f, rect, padding=True)


def hull_box(rect, widen=1e-9):
    """The smallest box that contains f(rect), widened by
    widen * (1 + max |end point|) in the first coordinate and, when widen > 0,
    by 1e-12 in the second (see the module docstring for the closed form).
    With widen = 0 it is the hull itself, up to the rounding of f."""
    x0, y0, x1, y1 = rect
    a, b = THETA
    v = [f1(x0, y0), f1(x0, y1), f1(x1, y0), f1(x1, y1)]
    lo, hi = min(v), max(v)
    for c in (y0, y1):
        xc = 10.0 - b / a * c
        if x0 < xc < x1:
            hi = max(hi, f1(xc, c))
    for c in (x0, x1):
        yc = 10.0 - a / b * c
        if y0 < yc < y1:
            hi = max(hi, f1(c, yc))
    e = widen * (1.0 + max(abs(lo), abs(hi)))
    ey = 1e-12 if widen > 0 else 0.0
    return [lo - e, 0.7 * x0 - ey, hi + e, 0.7 * x1 + ey]


def hull_box_map(rect):
    return hull_box(list(rect))


BOX_MAPS = {"corner": corner_box_map, "padded": padded_box_map, "hull": hull_box_map}


def leslie_model(subdiv_min, subdiv_max, init=None, box_map="corner",
                 lower=LOWER, upper=UPPER):
    """Model(min, max, lower, upper, F) when init is None (init 0, limit
    10000), else Model(min, max, init, 10000, lower, upper, F)."""
    CMGDB = _cmgdb()
    F = BOX_MAPS[box_map] if isinstance(box_map, str) else box_map
    if init is None:
        return CMGDB.Model(subdiv_min, subdiv_max, list(lower), list(upper), F)
    return CMGDB.Model(subdiv_min, subdiv_max, init, LIMIT, list(lower), list(upper), F)


# ---------------------------------------------------------------------------
# Boxes and grids
# ---------------------------------------------------------------------------

def depth_of(box, lower=LOWER, upper=UPPER):
    """Number of bisections that produce box from the phase space."""
    nx = round(math.log2((upper[0] - lower[0]) / (box[2] - box[0])))
    ny = round(math.log2((upper[1] - lower[1]) / (box[3] - box[1])))
    return int(nx + ny)


def ancestor_chain(point, depth, lower=LOWER, upper=UPPER):
    """Boxes of depth 0, ..., depth that contain point.  The k-th bisection
    halves along x when k is even and along y when k is odd, and a point on
    a bisecting line goes to the lower half."""
    box = [lower[0], lower[1], upper[0], upper[1]]
    chain = [list(box)]
    for k in range(depth):
        axis = k % 2
        mid = (box[axis] + box[axis + 2]) / 2
        if point[axis] <= mid:
            box[axis + 2] = mid
        else:
            box[axis] = mid
        chain.append(list(box))
    return chain


def parent_box(box, lower=LOWER, upper=UPPER):
    """The box of depth d - 1 that contains a box of depth d >= 1."""
    d = depth_of(box, lower, upper)
    x0, y0, x1, y1 = box
    if d % 2 == 1:             # the d-th bisection was along x
        w = x1 - x0
        k = round((x0 - lower[0]) / w)
        return [x0, y0, x0 + 2 * w, y1] if k % 2 == 0 else [x0 - w, y0, x1, y1]
    h = y1 - y0
    k = round((y0 - lower[1]) / h)
    return [x0, y0, x1, y0 + 2 * h] if k % 2 == 0 else [x0, y0 - h, x1, y1]


def contains_point(box, p):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def boxes_meet(a, b, tol=0.0):
    return a[0] <= b[2] + tol and b[0] <= a[2] + tol and a[1] <= b[3] + tol and b[1] <= a[3] + tol


def box_inside(inner, outer, rel=1e-8):
    """inner is contained in outer up to rel * (1 + max |end point|)."""
    t = rel * (1.0 + max(abs(v) for v in list(outer) + list(inner)))
    return (outer[0] <= inner[0] + t and outer[1] <= inner[1] + t
            and inner[2] <= outer[2] + t and inner[3] <= outer[3] + t)


def bounding_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def fmt_box(b, digits=4):
    return f"[{b[0]:.{digits}f}, {b[2]:.{digits}f}] x [{b[1]:.{digits}f}, {b[3]:.{digits}f}]"


def histogram(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def lattice_squares(boxes, depth, lower=LOWER, upper=UPPER):
    """The squares (i, j) of the uniform lattice of the given even depth
    whose union is the union of boxes (each box of depth <= depth)."""
    n = 2 ** (depth // 2)
    wx = (upper[0] - lower[0]) / n
    wy = (upper[1] - lower[1]) / n
    out = set()
    for x0, y0, x1, y1 in boxes:
        i0, i1 = round((x0 - lower[0]) / wx), round((x1 - lower[0]) / wx)
        j0, j1 = round((y0 - lower[1]) / wy), round((y1 - lower[1]) / wy)
        out.update((i, j) for i in range(i0, i1) for j in range(j0, j1))
    return out


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def strongly_connected_components(adj):
    """Tarjan's algorithm without recursion.  adj[v] lists the successors of v."""
    n = len(adj)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack, comps, counter = [], [], 0
    for root in range(n):
        if index[root] >= 0:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, iter(adj[root]))]
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if index[w] < 0:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(adj[w])))
                    advanced = True
                    break
                if on_stack[w] and index[w] < low[v]:
                    low[v] = index[w]
            if advanced:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                if low[v] < low[u]:
                    low[u] = low[v]
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
    return comps


def recurrent_components(adj):
    """Strongly connected components that contain a cycle (a self-loop counts)."""
    return [c for c in strongly_connected_components(adj)
            if len(c) > 1 or c[0] in set(adj[c[0]])]


def reachable(adj, seeds):
    seen = set(seeds)
    frontier = list(seeds)
    while frontier:
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return seen


# ---------------------------------------------------------------------------
# Homology of planar cubical sets over the field with 5 elements
# ---------------------------------------------------------------------------

P_FIELD = 5


def _faces(squares):
    S = set(squares)
    E, V = set(), set()
    for (i, j) in S:
        E.update([("h", i, j), ("h", i, j + 1), ("v", i, j), ("v", i + 1, j)])
        V.update([(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)])
    return S, E, V


def _rank_mod_p(M):
    import numpy as np
    M = M.copy() % P_FIELD
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), P_FIELD - 2, P_FIELD)) % P_FIELD
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        if len(others):
            M[others] = (M[others] - np.outer(M[others, c], M[r])) % P_FIELD
        r += 1
    return r


def betti_numbers(squares):
    """(b0, b1, b2) of a finite union of closed unit squares in the plane:
    b0 by union-find on squares that share a vertex, b1 = b0 - Euler
    characteristic, and b2 = 0."""
    sq = sorted(set(squares))
    if not sq:
        return (0, 0, 0)
    S, E, V = _faces(sq)
    parent = {s: s for s in sq}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    for (i, j) in sq:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                t = (i + di, j + dj)
                if t in parent:
                    ra, rb = find((i, j)), find(t)
                    if ra != rb:
                        parent[ra] = rb
    b0 = len({find(s) for s in sq})
    chi = len(V) - len(E) + len(S)
    return (b0, b0 - chi, 0)


def relative_homology_ranks(X_squares, A_squares):
    """Ranks of H_k(|X|, |A|) over F_5, k = 0, 1, 2, from the boundary matrices of
    the cubical chain complex of |X| modulo the faces of |A|."""
    import numpy as np
    _, EX, VX = _faces(X_squares)
    _, EA, VA = _faces(A_squares)
    S = sorted(set(X_squares) - set(A_squares))
    E = sorted(EX - EA)
    V = sorted(VX - VA)
    ei = {e: k for k, e in enumerate(E)}
    vi = {v: k for k, v in enumerate(V)}
    d2 = np.zeros((len(E), len(S)), dtype=np.int64)
    for k, (i, j) in enumerate(S):
        for e, sgn in ((("h", i, j), 1), (("v", i + 1, j), 1), (("h", i, j + 1), -1), (("v", i, j), -1)):
            if e in ei:
                d2[ei[e], k] = sgn % P_FIELD
    d1 = np.zeros((len(V), len(E)), dtype=np.int64)
    for k, (t, i, j) in enumerate(E):
        u, w = ((i, j), (i + 1, j)) if t == "h" else ((i, j), (i, j + 1))
        if w in vi:
            d1[vi[w], k] = 1
        if u in vi:
            d1[vi[u], k] = P_FIELD - 1
    assert not ((d1 @ d2) % P_FIELD).any()
    r1 = _rank_mod_p(d1) if d1.size else 0
    r2 = _rank_mod_p(d2) if d2.size else 0
    return (len(V) - r1, len(E) - r1 - r2, len(S) - r2)


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
    """Parse --tag (and the options added by extra(parser))."""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else None)
    parser.add_argument("--tag", default=None,
                        help="build tag appended to output names (default: cmgdb-<version>)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    if args.tag is None:
        args.tag = default_tag()
    return args


def header_lines(script, tag, cmgdb=True):
    """The header of an output.  It also lists the CMGDB_MAPGRAPH_* variables
    set in the environment, which change the number of box map evaluations.
    The stored outputs were computed with none of them set."""
    import numpy as np
    lines = [f"script: {Path(script).name}"]
    if cmgdb:
        lines.append(f"CMGDB version: {cmgdb_version()}   build tag: {tag}")
    lines.append(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
                 f"numpy {np.__version__}")
    mapgraph = sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith("CMGDB_MAPGRAPH_"))
    if cmgdb and mapgraph:
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
            json.dump(obj, fh, indent=None, separators=(",", ":"), sort_keys=False)
            fh.write("\n")

    def close(self):
        self._fh.close()


def load_json(name, tag=None):
    stem = f"{name}_{tag}" if tag else name
    with open(OUTPUTS / f"{stem}.json") as fh:
        return json.load(fh)
