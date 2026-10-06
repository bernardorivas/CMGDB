"""The chain lift of chomp's RelativeMapHomology, for complexes of dimension 1.

CMGDB computes the map that a combinatorial map F on the cubes of a complex
X' induces on H_1(X', A'; F_5) by lifting a relative cycle z through the
graph of F (chomp/RelativeMapHomology.h and chomp/FiberComplex.h).  On the
line, with cube k = [k, k + 1] and dk = (k + 1) - k:

  1. For each term a t of z, choose a vertex w_t of the closure of F(t).
     chomp takes the first element of a hash set of vertices.
  2. Over each end v of t, add a (dt : v) w_t to a 0-chain c_v.
  3. Over each vertex v, let Y_v be the closure of the union of F(Q) over the
     cubes Q of X' that contain v, and B_v the closure of the union of F(Q)
     over those that lie in A'.  Drop from c_v the vertices in B_v, and
     solve dp_v = c_v for a 1-chain p_v on the cubes of Y_v that are not in
     B_v, with the vertices in B_v dropped from dp_v.  With no solution,
     chomp reports the index as undefined (the fork release 1.3.3+fork.6
     raises RuntimeError instead).
  4. The image of z is -sum_v p_v, read in H_1(X', A') after dropping the
     cubes of A'.

If (X', A') is an index pair for F, that is F(A') cap X' subset A', the
image does not depend on the choices.  The function lift_outcomes computes
the image for every choice of the vertices w_t and every solution p_v, so it
yields the set of values that the lift can return.  This is a
re-implementation, compared with chomp's ComputeConleyIndex by
bounds_mechanism.py, not chomp itself.  It needs no CMGDB.
"""

import itertools

P = 5
POLY = {0: "0", 1: "x-1", 2: "x-2", 3: "x+2", 4: "x+1"}   # x - a over F_5, as CMGDB prints it


def _rref(rows, ncols):
    M = [[x % P for x in r] for r in rows]
    piv, r = [], 0
    for col in range(ncols):
        p = next((i for i in range(r, len(M)) if M[i][col]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = pow(M[r][col], P - 2, P)
        M[r] = [(x * inv) % P for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][col]:
                f = M[i][col]
                M[i] = [(x - f * y) % P for x, y in zip(M[i], M[r])]
        piv.append(col)
        r += 1
    return M[:r], piv


def solve(A, b, n):
    """Solutions of A x = b over F_5 (A given by rows over n unknowns), as
    (particular solution, basis of the kernel), or None."""
    if n == 0:
        return ([], []) if all(v % P == 0 for v in b) else None
    R, piv = _rref([list(A[i]) + [b[i]] for i in range(len(A))], n + 1)
    if n in piv:
        return None
    x0 = [0] * n
    for i, c in enumerate(piv):
        x0[c] = R[i][n]
    null = []
    for fj in (j for j in range(n) if j not in piv):
        x = [0] * n
        x[fj] = 1
        for i, c in enumerate(piv):
            x[c] = (-R[i][fj]) % P
        null.append(x)
    return x0, null


def _verts(cubes):
    return {v for k in cubes for v in (k, k + 1)}


def relative_homology(K, A):
    """Relative chains of (X', A') on the line (X' the cubes K, A' the cubes
    A): the cubes of K not in A' and the vertices of X' not in the closure of
    A'.  Returns (cubes, vertices, basis of H_1, dim H_0)."""
    cubes = sorted(set(K) - set(A))
    verts = sorted(_verts(K) - _verts(A))
    vi = {v: i for i, v in enumerate(verts)}
    rows = [[0] * len(cubes) for _ in verts]
    for j, k in enumerate(cubes):
        for v, s in ((k + 1, 1), (k, -1)):
            if v in vi:
                rows[vi[v]][j] = (rows[vi[v]][j] + s) % P
    sol = solve(rows, [0] * len(verts), len(cubes))
    basis = sol[1]
    rank = len(_rref(rows, len(cubes))[1]) if cubes and verts else 0
    return cubes, verts, basis, len(verts) - rank


def lift_outcomes(K, A, F, max_solutions=625):
    """For each basis cycle z of H_1(X', A'), the image of z under the lift,
    for every choice of vertices and of preboundaries.

    K, A: lists of cube offsets.  F: dict cube -> list of cubes.
    Returns a dict with the dimensions of H_0 and H_1, the basis cycles, for
    each cycle a list of (outcome, vertex choices), where an outcome is a
    tuple of coordinates in the basis or the text "no preboundary", and the
    largest dimension of the space of preboundaries met (0 when every
    preboundary that exists is unique)."""
    K = sorted(K)
    A = set(A)
    cubes, verts, basis, h0 = relative_homology(K, A)
    report = dict(H0_dim=h0, H1_dim=len(basis), cycles=[], outcomes=[], kernel_max=0)
    for b in basis:
        z = {cubes[i]: c for i, c in enumerate(b) if c}
        report["cycles"].append(sorted(z.items()))
        terms = sorted(z)
        choices = [sorted(_verts(F[t])) for t in terms]
        tally = {}
        for ws in itertools.product(*choices):
            chains = {}
            for t, w in zip(terms, ws):
                for v, s in ((t + 1, 1), (t, -1)):
                    c = chains.setdefault(v, {})
                    c[w] = (c.get(w, 0) + z[t] * s) % P
            partial = [{}]
            status = None
            for v in sorted(chains):
                nbs = [q for q in (v - 1, v) if q in K]
                Y = set().union(*[set(F[q]) for q in nbs])
                B = set().union(*[set(F[q]) for q in nbs if q in A])
                fib_cubes = sorted(Y - B)
                fib_verts = sorted(_verts(Y) - _verts(B))
                vi = {x: i for i, x in enumerate(fib_verts)}
                rows = [[0] * len(fib_cubes) for _ in fib_verts]
                for j, k in enumerate(fib_cubes):
                    for x, s in ((k + 1, 1), (k, -1)):
                        if x in vi:
                            rows[vi[x]][j] = (rows[vi[x]][j] + s) % P
                rhs = [chains[v].get(x, 0) for x in fib_verts]
                sol = solve(rows, rhs, len(fib_cubes))
                if sol is None:
                    status = "no preboundary"
                    break
                x0, null = sol
                report["kernel_max"] = max(report["kernel_max"], len(null))
                sols = []
                for coeffs in itertools.product(range(P), repeat=len(null)):
                    sols.append([(x0[i] + sum(c * nv[i] for c, nv in zip(coeffs, null))) % P
                                 for i in range(len(fib_cubes))])
                new = []
                for ans in partial:
                    for s in sols:
                        a2 = dict(ans)
                        for k, val in zip(fib_cubes, s):
                            if val:
                                a2[k] = (a2.get(k, 0) - val) % P
                        new.append(a2)
                        if len(new) > max_solutions:
                            raise RuntimeError("too many preboundaries to enumerate")
                partial = new
            if status:
                tally.setdefault(status, []).append(list(ws))
                continue
            for ans in partial:
                rel = [ans.get(k, 0) % P for k in cubes]
                # coordinates of rel in the basis (no 2-cells, so H_1 = cycles)
                cols = [[basis[j][i] for j in range(len(basis))] for i in range(len(cubes))]
                sol = solve(cols, rel, len(basis))
                key = "not a relative cycle" if sol is None else tuple(sol[0])
                lst = tally.setdefault(key, [])
                if list(ws) not in lst:
                    lst.append(list(ws))
        report["outcomes"].append(sorted(tally.items(), key=lambda kv: str(kv[0])))
    return report


def outcome_text(outcome):
    """Annotation in degree 1 for a 1 x 1 matrix (a), as CMGDB prints it."""
    if isinstance(outcome, tuple) and len(outcome) == 1:
        return POLY[outcome[0] % P]
    return str(outcome)
