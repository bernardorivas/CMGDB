"""Dynamics of the Leslie map f(x, y) = ((19.6 x + 23.68 y) e^(-0.1 (x + y)), 0.7 x).

Computes, without CMGDB:
  1. the fixed points 0 and p* and the eigenvalues and eigenvectors of Df there
     (closed form).
  2. a search for periodic orbits of minimal period <= 12 with all points in
     Q = [-0.001, 90] x [-0.001, 70], by Newton's method from a 91 x 71 grid of
     seeds (evidence, not a proof that the list is complete).
  3. the attracting 3-cycle a3 and the saddle 3-cycle s3 (Newton's method) with
     the eigenvalues of Df^3.
  4. the attracting invariant circle Gamma around p* (numerical evidence):
     extent, angle increments about p*, rotation number, star-shapedness about
     p*, no period up to 2000, Lyapunov exponents from two seeds.
  5. the basins of a3 and Gamma on a 361 x 281 grid of seeds.
  6. where the unstable manifolds of 0 and of s3 go, and the orbits of seeds
     near p*.
Also stores the data of figure 1 (Gamma, the branches of W^u(s3), the basins).

Output: outputs/leslie_dynamics.txt and outputs/leslie_dynamics.json.
Deterministic (no random numbers).  Runs in any of the venvs (numpy only),
in about 15 s on the Apple silicon machine of the stored outputs.

usage: python leslie_dynamics.py
"""

import math
import sys

sys.dont_write_bytecode = True

import numpy as np

import leslie_common as LC
from leslie_common import THETA, LOWER, UPPER, P_STAR, A3, S3, f, f_array

A, B = THETA


def jac_array(P):
    """Df at an (n, 2) array of points, as an (n, 2, 2) array."""
    x, y = P[:, 0], P[:, 1]
    e = np.exp(-0.1 * (x + y))
    s = A * x + B * y
    D = np.zeros((len(P), 2, 2))
    D[:, 0, 0] = (A - 0.1 * s) * e
    D[:, 0, 1] = (B - 0.1 * s) * e
    D[:, 1, 0] = 0.7
    return D


def newton_periodic_search(seeds, n, iters=80):
    """Damped Newton's method for f^n(z) = z from every seed at once.
    Returns the converged points (residual < 1e-9) that lie in Q."""
    P = seeds.astype(float).copy()
    alive = np.ones(len(P), bool)
    eye = np.eye(2)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            Z = P.copy()
            J = np.broadcast_to(eye, (len(P), 2, 2)).copy()
            for _ in range(n):
                J = jac_array(Z) @ J
                Z = f_array(Z)
            r = Z - P
            M = J - eye
            det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
            dx = -(M[:, 1, 1] * r[:, 0] - M[:, 0, 1] * r[:, 1]) / det
            dy = -(-M[:, 1, 0] * r[:, 0] + M[:, 0, 0] * r[:, 1]) / det
            step = np.hypot(dx, dy)
            scale = np.where(step > 10.0, 10.0 / step, 1.0)
            P = P + np.stack([dx * scale, dy * scale], axis=1)
            bad = ~np.isfinite(P).all(axis=1) | (np.abs(P) > 1e3).any(axis=1)
            alive &= ~bad
            P[~alive] = 0.0
        Z = P.copy()
        for _ in range(n):
            Z = f_array(Z)
        res = np.hypot(*(Z - P).T)
    inside = (P[:, 0] >= LOWER[0] - 1e-9) & (P[:, 0] <= UPPER[0] + 1e-9) & \
             (P[:, 1] >= LOWER[1] - 1e-9) & (P[:, 1] <= UPPER[1] + 1e-9)
    return P[alive & (res < 1e-9) & inside]


def orbit_of(z, k):
    out = [tuple(z)]
    for _ in range(k - 1):
        out.append(tuple(f(out[-1])))
    return out


def classify(l1, l2):
    m = sorted([abs(l1), abs(l2)])
    if m[1] < 1:
        return "attracting"
    if m[0] > 1:
        return "repelling"
    return "saddle"


def in_box(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def fates(P, steps, a3, rlo, rhi):
    """Classify where points go: 'a3', 'Gamma' (distance to p* within
    [rlo, rhi]), '0', 'escapes' (leaves through x < -1000 or overflows) or
    'other'."""
    P = np.array(P, float)
    with np.errstate(all="ignore"):
        for _ in range(steps):
            P = f_array(P)
            P[(P[:, 0] < -1e3) | ~np.isfinite(P).all(axis=1)] = np.nan
    finite = np.isfinite(P).all(axis=1)
    da = np.full(len(P), np.inf)
    for q in a3:
        da = np.minimum(da, np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]))
    dg = np.hypot(P[:, 0] - P_STAR[0], P[:, 1] - P_STAR[1])
    out = np.full(len(P), "other", dtype=object)
    out[finite & (dg >= rlo) & (dg <= rhi)] = "Gamma"
    out[finite & (da < 1e-6)] = "a3"
    out[finite & (np.hypot(P[:, 0], P[:, 1]) < 1e-12)] = "0"
    out[~finite] = "escapes"
    return out


def count(labels):
    vals, cnt = np.unique(labels.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnt)}


def main():
    LC.parse_args(__doc__)          # accepts --tag like the other scripts, and does not use it
    out = LC.Output("leslie_dynamics")
    out(*LC.header_lines(__file__, None, cmgdb=False))
    rec = {}
    L = [LOWER[0], LOWER[1], (LOWER[0] + UPPER[0]) / 2, UPPER[1]]
    C = [(LOWER[0] + UPPER[0]) / 2, LOWER[1], UPPER[0], UPPER[1]]
    Cp = [LOWER[0], (LOWER[1] + UPPER[1]) / 2, (LOWER[0] + UPPER[0]) / 2, UPPER[1]]
    out(f"Q = [{LOWER[0]}, {UPPER[0]}] x [{LOWER[1]}, {UPPER[1]}];  L = {LC.fmt_box(L)},  C = {LC.fmt_box(C)},  "
        f"C' = {LC.fmt_box(Cp)}")

    # 1. Fixed points ---------------------------------------------------------
    out("", "1. Fixed points.  f(x, y) = (x, y) iff y = 0.7 x and x = (a + 0.7 b) x exp(-0.17 x), "
        f"a + 0.7 b = {A + 0.7 * B:.6f}:")
    out(f"   (0, 0) and p* = (ln(a + 0.7 b)/0.17, 0.7 ln(a + 0.7 b)/0.17) = ({P_STAR[0]:.6f}, {P_STAR[1]:.6f}); "
        f"|f(p*) - p*| = {math.hypot(f(P_STAR)[0] - P_STAR[0], f(P_STAR)[1] - P_STAR[1]):.1e}")
    fixed = {}
    for name, p in (("0", (0.0, 0.0)), ("p*", P_STAR)):
        J = LC.Df(*p)
        l1, l2, t, d = LC.eig2(J)
        fixed[name] = {"point": list(p), "Df": J, "eigenvalues": [[l1.real, l1.imag], [l2.real, l2.imag]],
                       "moduli": [abs(l1), abs(l2)], "trace": t, "det": d}
        out(f"   {name}: Df = [[{J[0][0]:.6f}, {J[0][1]:.6f}], [{J[1][0]:.1f}, {J[1][1]:.1f}]], trace {t:.6f}, det {d:.6f}; "
            f"eigenvalues {l1:.6f}, {l2:.6f}; moduli {abs(l1):.6f}, {abs(l2):.6f}; {classify(l1, l2)}")
    lu = (A + math.sqrt(A * A + 2.8 * B)) / 2
    ls = (A - math.sqrt(A * A + 2.8 * B)) / 2
    vu, vs = (1.0, 0.7 / lu), (1.0, 0.7 / ls)
    fixed["0"]["unstable_vector"], fixed["0"]["stable_vector"] = list(vu), list(vs)
    out(f"   at 0 the eigenvector for lambda is (lambda, 0.7): unstable lambda_u = {lu:.6f}, direction (1, {vu[1]:.6f}); "
        f"stable lambda_s = {ls:.6f}, direction (1, {vs[1]:.6f})")
    rec["fixed_points"] = fixed

    # 2. Periodic orbits --------------------------------------------------------
    seeds = np.array([(x, y) for x in np.linspace(0.0, 90.0, 91) for y in np.linspace(0.0, 70.0, 71)])
    found = {}
    for n in range(1, 13):
        for z in newton_periodic_search(seeds, n):
            k = next(k for k in range(1, n + 1) if n % k == 0 and
                     np.hypot(*(np.array(LC.iterate_with_jacobian(z, k)[0]) - z)) < 1e-7)
            orb = orbit_of(z, k)
            if not all(in_box(q, [LOWER[0] - 1e-9, LOWER[1] - 1e-9, UPPER[0] + 1e-9, UPPER[1] + 1e-9]) for q in orb):
                continue
            key = (k, min((round(q[0], 5), round(q[1], 5)) for q in orb))
            found.setdefault(key, orb)
    out("", f"2. Periodic orbits of minimal period <= 12 with all points in Q, by Newton's method for f^n(z) = z, "
        f"n = 1, ..., 12, from {len(seeds)} seeds (a 91 x 71 grid of [0, 90] x [0, 70]): {len(found)} orbits")
    rec["periodic_orbit_search"] = {"seeds": len(seeds), "max_period": 12, "orbits": []}
    for (k, _), orb in sorted(found.items()):
        orb, res = LC.periodic_orbit(orb[0], k)
        _, J = LC.iterate_with_jacobian(orb[0], k)
        l1, l2, t, d = LC.eig2(J)
        out(f"   period {k}: {', '.join(f'({q[0]:.6f}, {q[1]:.6f})' for q in orb)};  eigenvalues of Df^{k}: "
            f"{l1:.6g}, {l2:.6g};  {classify(l1, l2)};  residual {res:.1e}")
        rec["periodic_orbit_search"]["orbits"].append({"period": k, "orbit": [list(q) for q in orb],
                                                         "eigenvalues": [[l1.real, l1.imag], [l2.real, l2.imag]],
                                                         "type": classify(l1, l2)})

    # 3. The two 3-cycles --------------------------------------------------------
    out("", "3. The 3-cycles (Newton's method for f^3(z) = z from (7.5, 0.5) and (19.2, 1.3)):")
    cycles = {}
    for name, orb, res in (("a3", A3, LC.A3_RESIDUAL), ("s3", S3, LC.S3_RESIDUAL)):
        _, J = LC.iterate_with_jacobian(orb[0], 3)
        l1, l2, t, d = LC.eig2(J)
        where = ["C" if in_box(q, C) else "C'" if in_box(q, Cp) else "L minus C'" for q in orb]
        cycles[name] = {"orbit": [list(q) for q in orb], "residual": res,
                        "eigenvalues_Df3": [l1.real, l2.real], "det_Df3": d, "type": classify(l1, l2),
                        "cells_depth_one": where}
        out(f"   {name} = {', '.join(f'({q[0]:.6f}, {q[1]:.6f})' for q in orb)};  |f^3(z) - z| = {res:.1e};  "
            f"eigenvalues of Df^3: {l1.real:.6f}, {l2.real:.6f} (imaginary parts {abs(l1.imag):.0e});  "
            f"det {d:.6f};  {classify(l1, l2)}")
        out(f"      the three points lie in: {', '.join(where)}")
    rec["three_cycles"] = cycles
    out("   the unstable eigenvalue of Df^3 along s3 is positive: "
        f"{cycles['s3']['eigenvalues_Df3'][0] > 1}")

    # 4. The invariant circle Gamma ------------------------------------------------
    N = 200000
    orb = np.array(LC.gamma_orbit(N, burn=100000, seed=(30.0, 20.0)))
    d = orb - np.array(P_STAR)
    th = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    dth = np.mod(np.diff(th), 2 * np.pi)
    rho = dth.sum() / (2 * np.pi * len(dth))
    idx = np.argsort(th)
    ts, rs, ps = th[idx], r[idx], orb[idx]
    gaps = np.hypot(*np.diff(np.vstack([ps, ps[:1]]), axis=0).T)
    bins = np.floor((th + np.pi) / (2 * np.pi) * 4000).astype(int)
    spread = max(r[bins == k].max() - r[bins == k].min() for k in np.unique(bins))
    period = next((p for p in range(1, 2001) if np.max(np.abs(orb[p:p + 50] - orb[:50])) < 1e-6), None)
    orb2 = np.array(LC.gamma_orbit(N, burn=100000, seed=(25.0, 12.0)))
    d2 = orb2 - np.array(P_STAR)
    r2 = np.hypot(d2[:, 0], d2[:, 1])
    rho2 = np.mod(np.diff(np.arctan2(d2[:, 1], d2[:, 0])), 2 * np.pi).sum() / (2 * np.pi * (N - 1))

    def lyapunov(seed, n=200000, burn=20000):
        x, y = seed
        for _ in range(burn):
            x, y = f((x, y))
        u = (1.0, 0.0)
        s1 = s12 = 0.0
        for _ in range(n):
            J = LC.Df(x, y)
            u = (J[0][0] * u[0] + J[0][1] * u[1], J[1][0] * u[0] + J[1][1] * u[1])
            nu = math.hypot(*u)
            s1 += math.log(nu)
            u = (u[0] / nu, u[1] / nu)
            s12 += math.log(abs(J[0][0] * J[1][1] - J[0][1] * J[1][0]))
            x, y = f((x, y))
        return s1 / n, s12 / n - s1 / n
    lyap = {str(s): lyapunov(s) for s in ((30.0, 20.0), (21.2, 14.8))}
    out("", f"4. Gamma: orbit of (30, 20), {N} points after 100000 steps.")
    out(f"   x in [{orb[:, 0].min():.4f}, {orb[:, 0].max():.4f}], y in [{orb[:, 1].min():.4f}, {orb[:, 1].max():.4f}], "
        f"|z - p*| in [{r.min():.4f}, {r.max():.4f}]")
    out(f"   angle about p* advances by {dth.min():.5f} to {dth.max():.5f} rad per step (mod 2 pi); "
        f"rotation number {rho:.6f} turns per step")
    out(f"   points sorted by angle: largest angle gap {np.max(np.diff(np.concatenate([ts, ts[:1] + 2 * np.pi]))):.2e} rad, "
        f"largest distance between consecutive points {gaps.max():.4f}, largest radius jump {np.max(np.abs(np.diff(rs))):.4f}; "
        f"largest radius spread in 4000 equal angle sectors {spread:.4f}")
    out(f"   period up to 2000 (tolerance 1e-6): {period if period else 'none'}")
    out(f"   orbit of (25, 12): |z - p*| in [{r2.min():.4f}, {r2.max():.4f}], rotation number {rho2:.6f}")
    for s, (l1, l2) in lyap.items():
        out(f"   Lyapunov exponents from {s} (200000 steps after 20000): {l1:.2e}, {l2:.5f}")
    rec["gamma"] = {"x_range": [orb[:, 0].min(), orb[:, 0].max()], "y_range": [orb[:, 1].min(), orb[:, 1].max()],
                    "radius_range": [r.min(), r.max()], "angle_increment_range": [dth.min(), dth.max()],
                    "rotation_number": rho, "max_consecutive_gap_sorted": gaps.max(),
                    "max_radius_jump_sorted": float(np.max(np.abs(np.diff(rs)))), "max_radius_spread_4000_sectors": spread,
                    "period_up_to_2000": period, "second_seed_radius_range": [r2.min(), r2.max()],
                    "second_seed_rotation_number": rho2,
                    "lyapunov": {s: list(v) for s, v in lyap.items()}}
    rlo, rhi = r.min() - 1e-3, r.max() + 1e-3
    # the curve for the figures: 4000 points sorted by angle
    sel = idx[:: max(1, len(idx) // 4000)]
    rec["gamma"]["curve"] = orb[sel].round(6).tolist()

    # 5. Basins -------------------------------------------------------------------
    xs, ys = np.linspace(0.0, 90.0, 361), np.linspace(0.0, 70.0, 281)
    X, Y = np.meshgrid(xs, ys)
    lab = fates(np.stack([X.ravel(), Y.ravel()], axis=1), 5000, A3, rlo, rhi)
    cnt = count(lab)
    out("", f"5. Basins: {lab.size} seeds on a 361 x 281 grid of [0, 90] x [0, 70] (spacing 0.25), 5000 steps: {cnt}")
    code = {"a3": "1", "Gamma": "2", "0": "0", "other": "9", "escapes": "8"}
    rec["basins"] = {"xs": [0.0, 90.0, 361], "ys": [0.0, 70.0, 281], "steps": 5000, "counts": cnt,
                     "codes": {"1": "a3", "2": "Gamma", "0": "0", "9": "other", "8": "escapes"},
                     "rows": ["".join(code[v] for v in row) for row in lab.reshape(Y.shape)]}

    # 6. Connections ----------------------------------------------------------------
    out("", "6. Connections (fates after 10000 steps).")
    v = np.array(vu) / np.hypot(*vu)
    conn = {}
    for t0, m in ((1e-7, 20001), (1e-9, 30001)):
        t = np.geomspace(t0, lu * t0, m)
        c = count(fates(t[:, None] * v[None], 10000, A3, rlo, rhi))
        conn[f"Wu0_quadrant_{t0:g}"] = c
        out(f"   W^u(0), branch into the quadrant, {m} points t v_u, t in [{t0:g}, lambda_u {t0:g}]: {c}")
    t = np.geomspace(1e-9, lu * 1e-9, 200)
    c = count(fates(-t[:, None] * v[None], 50, A3, rlo, rhi))
    conn["Wu0_other_branch"] = c
    out(f"   W^u(0), the other branch -t v_u, 200 points, 50 steps: {c}")
    curves = []
    for k, q in enumerate(S3):
        _, J = LC.iterate_with_jacobian(q, 3)
        w, V = np.linalg.eig(np.array(J))
        i = int(np.argmax(np.abs(w)))
        vu3, mu = np.real(V[:, i]), float(np.real(w[i]))
        for sgn in (1, -1):
            eps = np.geomspace(1e-6, mu * 1e-6, 2001)
            c = count(fates(np.array(q)[None] + sgn * eps[:, None] * vu3[None], 10000, A3, rlo, rhi))
            conn[f"Wu_s3[{k}]_{'+' if sgn > 0 else '-'}"] = c
            out(f"   W^u(s3[{k}]), branch {'+' if sgn > 0 else '-'}(unstable eigenvector {vu3.round(4).tolist()}), "
                f"2001 points: {c}")
            # the branch as a curve: 18 images of a fundamental domain under f^3,
            # keeping points at least 0.05 apart
            P = np.array(q)[None] + sgn * np.geomspace(1e-4, mu * 1e-4, 400)[:, None] * vu3[None]
            pts = []
            for _ in range(18):
                pts.extend(P.tolist())
                P = f_array(f_array(f_array(P)))
            kept = [pts[0]]
            for z in pts[1:]:
                if math.hypot(z[0] - kept[-1][0], z[1] - kept[-1][1]) > 0.05:
                    kept.append(z)
            curves.append({"point": k, "sign": sgn, "fate": max(c, key=c.get), "curve": np.round(kept, 4).tolist()})
    ring = {}
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    for rad in (1e-6, 1e-3, 0.1, 1.0, 3.0):
        P = np.array(P_STAR) + rad * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        lab = fates(P, 30000, A3, rlo, rhi)
        ring[str(rad)] = count(lab)
    out(f"   360 points on each circle of radius 1e-6, 1e-3, 0.1, 1, 3 about p*, 30000 steps: {ring}")
    conn["p*_circles"] = ring
    rec["connections"] = conn
    rec["unstable_branches_s3"] = curves
    rec["regions"] = {"L": L, "C": C, "C'": Cp}
    out.json(rec)
    out.close()


if __name__ == "__main__":
    sys.exit(main())
