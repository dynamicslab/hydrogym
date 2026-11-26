"""
Time- and span-averaged flow statistics of a deploy.py wing run from its restart fields (chk<actions>0.f00000,
one every --chunk actions of deploy.py), the input of bl_report.py:

    python bl_stats.py runs/zero_trip60 --skip 500             # -> runs/zero_trip60/field_stats.npz
    python bl_stats.py runs/zero_ref1 --skip 2420 --workers 16

The mesh is an extrusion in z with a periodic span, so all statistics are averages over z and over the fields at
each (x, y) position of the GLL points ("2D points"). Accumulated per 2D point are the raw moments of u_i up to
fourth order, p, p^2, p u_i, the velocity and pressure gradients (spectral-element derivatives per element) and
their products for all terms of the Reynolds-stress budgets (dissipation, velocity-pressure gradient, u_i u.grad u_j
for turbulent transport and convection, u_i lap u_j for the viscous diffusion): every derivative is the exact one of
the element polynomials of an instantaneous field, none of averaged fields.
Interpolated (linear on a Delaunay triangulation of the 2D points) and rotated into the local wall frame
(s downstream tangent, n wall normal into the fluid, z span) are:
  - wall-normal lines at --stations on both surfaces (profiles, boundary-layer integrals; also per field, for the
    confidence intervals), and wake lines at --wake (x frame);
  - instantaneous z-lines: u on the lines at --spectra-x (spanwise spectra and correlations), the wall shear and
    pressure at every wall station (Cf,rms, intermittency), u_s on wall-parallel curves at --curve-n (streamwise
    correlations, Euu(kx)).
The fields are 250 actions = 0.15 c/U_inf apart: ensemble averages over few, nearly independent fields and many
spanwise points, without temporal information (time series: the walls_*.npz of deploy.py).
Memory about 2.5 GB per worker plus 4 GB; about 4 min per N = 7 field and worker.
"""

import argparse
import itertools
import json
import multiprocessing
import os
import time

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, cKDTree

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--skip", type=int, default=0, help="use the fields written after this many actions")
p.add_argument("--workers", type=int, default=8)
p.add_argument(
    "--stations", type=float, nargs=3, default=[0.02, 0.98, 0.01], help="x/c from, to, step: wall-normal lines"
)
p.add_argument("--n-max", type=float, default=0.1, help="length of the wall-normal lines (c)")
p.add_argument("--n-points", type=int, default=240, help="points per wall-normal line")
p.add_argument(
    "--wake", type=float, nargs="+", default=[1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0], help="x/c of the wake lines"
)
p.add_argument("--wake-y", type=float, default=0.3, help="wake lines from -this to this (c)")
p.add_argument("--spectra-x", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
p.add_argument(
    "--curve-n",
    type=float,
    nargs="+",
    default=[0.002, 0.005, 0.01],
    help="wall distances (c) of the wall-parallel curves",
)
p.add_argument("--curve-x", type=float, nargs=3, default=[0.12, 0.98, 0.001], help="x/c from, to, step of the curves")
p.add_argument("--nu", type=float, default=1 / 200000)
p.add_argument("--chunk-elements", type=int, default=4000)
p.add_argument("--max-fields", type=int, default=0, help="use at most this many fields (0: all; tests)")
p.add_argument("--out", help="default RUN/field_stats.npz")
p.add_argument(
    "--int-mesh", help="NekStatsPy int_mesh.mat: also write RUN/ZSTAT/int_fld (KTH pstat2D format) at its points"
)
args = p.parse_args()
out_file = args.out or os.path.join(args.run, "field_stats.npz")
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.0f} s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- field files
state = json.load(open(os.path.join(args.run, "state.json")))
chunks = [c for c in state["chunks"] if c["actions"] > args.skip]
if args.max_fields:
    chunks = chunks[: args.max_fields]
files = [os.path.join(args.run, c["file"]) for c in chunks]
missing = [f for f in files if not os.path.exists(f)]
if missing or not files:
    raise SystemExit(f"{args.run}: no fields after {args.skip} actions" if not files else f"missing: {missing}")


def header(path):
    h = open(path, "rb").read(132).decode().split()
    r = {"wdsz": int(h[1]), "lx": int(h[2]), "nel": int(h[5]), "nelgt": int(h[6]), "time": float(h[7]), "fields": h[11]}
    if r["nel"] != r["nelgt"] or not r["fields"].startswith("XUP"):
        raise SystemExit(f"{path}: expected one file with XUP, got {h}")
    return r


def open_field(path):
    """Nek5000 field file (#std, one file): memmaps X, U (nel, 3, Np) and P (nel, Np)."""
    h = header(path)
    dt = "<f8" if h["wdsz"] == 8 else "<f4"
    nel, npt, w = h["nel"], h["lx"] ** 3, h["wdsz"]
    off = 136 + 4 * nel
    X = np.memmap(path, dt, "r", offset=off, shape=(nel, 3, npt))
    U = np.memmap(path, dt, "r", offset=off + nel * 3 * npt * w, shape=(nel, 3, npt))
    P = np.memmap(path, dt, "r", offset=off + nel * 6 * npt * w, shape=(nel, npt))
    return X, U, P


def group_ids(v, tol):
    """Labels of the values of v that agree within tol (sorted ascending)."""
    o = np.argsort(v, kind="stable")
    brk = np.ones(len(v), dtype=bool)
    brk[1:] = np.diff(v[o]) > tol
    lab = np.empty(len(v), dtype=np.int64)
    lab[o] = np.cumsum(brk) - 1
    return lab


def gll(n):
    """GLL points of order n and the derivative matrix D[i, j] = l_j'(x_i)."""
    from numpy.polynomial import legendre as L

    c = np.zeros(n + 1)
    c[n] = 1
    x = np.concatenate([[-1.0], np.sort(L.legroots(L.legder(c)).real), [1.0]])
    pn = L.legval(x, c)
    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                D[i, j] = pn[i] / (pn[j] * (x[i] - x[j]))
    D[0, 0], D[n, n] = -n * (n + 1) / 4, n * (n + 1) / 4
    return x, D


h0 = header(files[0])
LX, NEL = h0["lx"], h0["nel"]
NPT = LX**3
_, D = gll(LX - 1)
NU = args.nu


def dref(f):
    """Reference derivatives (d/dr, d/ds, d/dt) of f (ne, lx, lx, lx) in Nek order [k, j, i]."""
    return f @ D.T, D @ f, np.einsum("ck,ekji->ecji", D, f)


def jacobian_inv(xc):
    """xc (ne, 3, Np) -> Jinv (ne, Np, 3, 3), Jinv[..., q, m] = d r_q / d x_m."""
    ne = len(xc)
    J = np.empty((ne, NPT, 3, 3))
    for m in range(3):
        for q, d in enumerate(dref(xc[:, m].reshape(ne, LX, LX, LX))):
            J[:, :, m, q] = d.reshape(ne, NPT)
    return np.linalg.inv(J)


def grad(f, Jinv):
    """f (ne, Np) -> (ne * Np, 3) physical gradient."""
    ne = len(f)
    fr = np.stack([d.reshape(ne, NPT) for d in dref(f.reshape(ne, LX, LX, LX))], axis=-1)
    return np.einsum("epq,epqm->epm", fr, Jinv).reshape(-1, 3)


# ---------------------------------------------------------------------------------------- 2D points (groups)
log(f"{args.run}: {len(files)} fields after {args.skip} actions, {NEL} elements, N = {LX - 1}")
X0, _, _ = open_field(files[0])
xyz = np.asarray(X0).transpose(0, 2, 1).reshape(-1, 3)
tol = 1e-7
lx_ = group_ids(xyz[:, 0], tol)
# split by x, then by y within each x group (as layout.group_ids)
LAB = np.zeros(len(xyz), dtype=np.int64)
o = np.lexsort((xyz[:, 1], lx_))
brk = np.ones(len(o), dtype=bool)
brk[1:] = (lx_[o][1:] != lx_[o][:-1]) | (np.diff(xyz[o, 1]) > tol)
LAB[o] = np.cumsum(brk) - 1
del lx_, o, brk
LAB = LAB.astype(np.int32)
NG = int(LAB.max()) + 1
CNT = np.bincount(LAB, minlength=NG).astype(float)
GX = np.bincount(LAB, xyz[:, 0], NG) / CNT
GY = np.bincount(LAB, xyz[:, 1], NG) / CNT
zlev = group_ids(xyz[:, 2], tol)
zvals = np.bincount(zlev, xyz[:, 2]) / np.bincount(zlev)
Lz = float(zvals[-1] - zvals[0])
NZ = len(zvals) - 1  # periodic: the last level is the first
ZID = np.where(zlev == NZ, 0, zlev).astype(np.int16)
ZS = zvals[:-1] - zvals[0]
if not np.all(np.bincount(LAB.astype(np.int64) * NZ + ZID, minlength=NG * NZ) > 0):
    raise SystemExit("the mesh is not an extrusion in z: (x, y) points without all z levels")
log(f"{NG} 2D points, {NZ} z levels, span {Lz:.6g}")
del xyz, zlev

# ------------------------------------------------------------------------------------------- wall stations
geo = dict(np.load(os.path.join(args.run, "walls_geometry.npz")))
bids = [b for b in (1, 2) if f"x_b{b}" in geo]
SX = np.concatenate([geo[f"x_b{b}"] for b in bids])
SY = np.concatenate([geo[f"y_b{b}"] for b in bids])
SN = np.stack([np.concatenate([geo[f"nx_b{b}"] for b in bids]), np.concatenate([geo[f"ny_b{b}"] for b in bids])], 1)
SN /= np.linalg.norm(SN, axis=1, keepdims=True)  # into the fluid
ST = np.stack([SN[:, 1], -SN[:, 0]], 1)
ST *= np.sign(ST[:, 0] + 1e-300)[:, None]  # downstream tangent
tree_g = cKDTree(np.c_[GX, GY])
dist, WALLG = tree_g.query(np.c_[SX, SY])
if dist.max() > 1e-6:
    log(f"WARNING: wall station {dist.argmax()} is {dist.max():.2e} from the nearest 2D point")
NW = len(SX)
WROW = np.full(NG, -1, dtype=np.int32)
WROW[WALLG] = np.arange(NW)
# first off-wall point and station spacing (resolution)
dist_s, sid = cKDTree(np.c_[SX, SY]).query(np.c_[GX, GY])
dn = (GX - SX[sid]) * SN[sid, 0] + (GY - SY[sid]) * SN[sid, 1]
dn1 = np.full(NW, np.inf)
m = (dn > 1e-9) & (dist_s < 0.01)
np.minimum.at(dn1, sid[m], dn[m])
ds_wall = np.full(NW, np.nan)
for side in (1, -1):
    k = np.flatnonzero(np.sign(SY) == side)
    k = k[np.argsort(SX[k])]
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(SX[k]), np.diff(SY[k])))])
    ds_wall[k] = np.gradient(s)


def surface(x0, side):
    """Point, normal and tangent of the surface side (+1 upper, -1 lower) at x0, interpolated between stations."""
    k = np.flatnonzero(np.sign(SY) == side)
    k = k[np.argsort(SX[k])]
    xs = SX[k]
    pt = np.array([x0, np.interp(x0, xs, SY[k])])
    n = np.array([np.interp(x0, xs, SN[k, 0]), np.interp(x0, xs, SN[k, 1])])
    n /= np.linalg.norm(n)
    t = np.array([n[1], -n[0]])
    t *= np.sign(t[0] + 1e-300)  # downstream on both sides
    return pt, n, t


# ------------------------------------------------------------------------------------------ lines, curves
xst = np.arange(args.stations[0], args.stations[1] + 1e-9, args.stations[2])
nvals = np.concatenate([[0.0], np.geomspace(2e-6, args.n_max, args.n_points - 1)])
lines = []  # (kind, x0, side, origin, t, n): points origin + nvals * n
for side in (1, -1):
    for x0 in xst:
        o_, n_, t_ = surface(x0, side)
        lines.append(("wall", x0, side, o_, t_, n_))
yw = np.linspace(-args.wake_y, args.wake_y, 601)
for x0 in args.wake:
    lines.append(("wake", x0, 0, np.array([x0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])))
NL_WALL = 2 * len(xst)
pts, frames, line_of = [], [], []
for i, (kind, x0, side, o_, t_, n_) in enumerate(lines):
    if kind == "wall":
        q = o_[None] + nvals[:, None] * n_[None]
    else:
        q = np.c_[np.full(len(yw), x0), yw]
    pts.append(q)
    frames.append(np.repeat(np.r_[t_, n_][None], len(q), 0))
    line_of.append(np.full(len(q), i))
pts, frames, line_of = np.concatenate(pts), np.concatenate(frames), np.concatenate(line_of)

log("Delaunay triangulation of the 2D points")
tri = Delaunay(np.c_[GX, GY])


def interp_matrix(q, wall_rows=None):
    """Sparse (len(q), NG) linear interpolation; rows in wall_rows take the nearest 2D point (the wall node)."""
    s = tri.find_simplex(q)
    ok = s >= 0
    if wall_rows is not None:
        ok[wall_rows] = False
    T = tri.transform[s[ok]]
    b = np.einsum("nij,nj->ni", T[:, :2], q[ok] - T[:, 2])
    rows = [np.repeat(np.flatnonzero(ok), 3)]
    cols = [tri.simplices[s[ok]].ravel()]
    vals = [np.c_[b, 1 - b.sum(1)].ravel()]
    if wall_rows is not None:
        rows.append(wall_rows)
        cols.append(tree_g.query(q[wall_rows])[1])
        vals.append(np.ones(len(wall_rows)))
    W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(q), NG))
    valid = ok.copy()
    if wall_rows is not None:
        valid[wall_rows] = True
    return W, valid


is_wall_pt = np.zeros(len(pts), dtype=bool)
start = 0
for i, (kind, *_r) in enumerate(lines):
    n_i = (line_of == i).sum()
    if kind == "wall":
        is_wall_pt[start] = True
    start += n_i
W_LINES, valid = interp_matrix(pts, np.flatnonzero(is_wall_pt))
log(f"{len(lines)} lines, {len(pts)} points ({(~valid).sum()} outside the mesh)")

# spectra: all off-wall points of the wall lines nearest to --spectra-x
spec_lines = [
    i for i, l in enumerate(lines[:NL_WALL]) if np.min(np.abs(np.array(args.spectra_x) - l[1])) < args.stations[2] / 2
]
spec_pts = np.flatnonzero(np.isin(line_of, spec_lines) & ~is_wall_pt & valid)
# wall-parallel curves
xc = np.arange(args.curve_x[0], args.curve_x[1] + 1e-9, args.curve_x[2])
curve_pts, curve_t = [], []
for side in (1, -1):
    for nc in args.curve_n:
        for x0 in xc:
            o_, n_, t_ = surface(x0, side)
            curve_pts.append(o_ + nc * n_)
            curve_t.append(t_)
curve_pts, curve_t = np.array(curve_pts), np.array(curve_t)
W_CURVE, cvalid = interp_matrix(curve_pts)
W_SPEC = W_LINES[spec_pts]
VCOLS = np.unique(np.concatenate([W_SPEC.indices, W_CURVE.indices]))
VROW = np.full(NG, -1, dtype=np.int32)
VROW[VCOLS] = np.arange(len(VCOLS))
W_SPEC_V, W_CURVE_V = W_SPEC[:, VCOLS], W_CURVE[:, VCOLS]

# ----------------------------------------------------------------------------------- accumulated quantities
C2 = list(itertools.combinations_with_replacement(range(3), 2))
C3 = list(itertools.combinations_with_replacement(range(3), 3))
C4 = list(itertools.combinations_with_replacement(range(3), 4))
NAMES = (
    ["u0", "u1", "u2", "p"]
    + [f"uu{a}{b}" for a, b in C2]
    + [f"uuu{''.join(map(str, c))}" for c in C3]
    + [f"uuuu{''.join(map(str, c))}" for c in C4]
    + ["pp", "pu0", "pu1", "pu2"]
    + [f"G{i}{k}" for i in range(3) for k in range(3)]
    + ["dp0", "dp1", "dp2"]
    + [f"GG{a}{b}" for a, b in C2]
    + [f"pG{i}{j}" for i in range(3) for j in range(3)]
    + [f"udp{i}{j}" for i in range(3) for j in range(3)]
    + [f"uG{i}{j}{k}" for i in range(3) for j in range(3) for k in range(3)]
    + [f"uA{j}{i}" for j in range(3) for i in range(3)]
    + ["A0", "A1", "A2", "L0", "L1", "L2"]
    + [f"uL{i}{j}" for i in range(3) for j in range(3)]
)
IDX = {n: k for k, n in enumerate(NAMES)}
NQ = len(NAMES)


def accumulate(path):
    """Sums over the points of every 2D point of all NAMES, and the z-line samples, of one field."""
    X, U, P = open_field(path)
    sums = np.zeros((NQ, NG))
    Bv = np.zeros((len(VCOLS), NZ, 3))
    Bw = np.zeros((NW, NZ, 3))
    CH = args.chunk_elements
    for e0 in range(0, NEL, CH):
        e1 = min(NEL, e0 + CH)
        Jinv = jacobian_inv(np.asarray(X[e0:e1], dtype=float))
        uc = np.asarray(U[e0:e1], dtype=float)
        pc = np.asarray(P[e0:e1], dtype=float)
        lab = LAB[e0 * NPT : e1 * NPT]
        zid = ZID[e0 * NPT : e1 * NPT]
        u = [uc[:, i].reshape(-1) for i in range(3)]
        pr = pc.reshape(-1)
        G = [grad(uc[:, i], Jinv) for i in range(3)]  # G[i][:, k] = d u_i / d x_k
        dp = grad(pc, Jinv)
        k = 0

        def add(v):
            nonlocal k
            sums[k] += np.bincount(lab, v, NG)
            k += 1

        for v in u + [pr]:
            add(v)
        for c in C2 + C3 + C4:
            v = u[c[0]].copy()
            for i in c[1:]:
                v *= u[i]
            add(v)
        add(pr * pr)
        for i in range(3):
            add(pr * u[i])
        for i in range(3):
            for kk in range(3):
                add(G[i][:, kk])
        for kk in range(3):
            add(dp[:, kk])
        for a, b in C2:
            add((G[a] * G[b]).sum(1))
        for i in range(3):
            for j in range(3):
                add(pr * G[i][:, j])
        for i in range(3):
            for j in range(3):
                add(u[i] * dp[:, j])
        for i in range(3):
            for j in range(3):
                for kk in range(3):
                    add(u[i] * G[j][:, kk])
        A = [sum(u[kk] * G[i][:, kk] for kk in range(3)) for i in range(3)]
        for j in range(3):
            for i in range(3):
                add(u[j] * A[i])
        for i in range(3):
            add(A[i])
        lap = [sum(grad(G[i][:, kk].reshape(e1 - e0, NPT), Jinv)[:, kk] for kk in range(3)) for i in range(3)]
        for i in range(3):
            add(lap[i])
        for i in range(3):
            for j in range(3):
                add(u[i] * lap[j])
        assert k == NQ
        # z-lines
        r = VROW[lab]
        m = r >= 0
        if m.any():
            for c in range(3):
                Bv[r[m], zid[m], c] = u[c][m]
        r = WROW[lab]
        m = r >= 0
        if m.any():
            g = r[m]
            n3 = np.c_[SN[g], np.zeros(len(g))]
            t3 = np.c_[ST[g], np.zeros(len(g))]
            Gm = np.stack([G[i][m] for i in range(3)], axis=1)  # (n, i, k)
            trac = NU * np.einsum("nik,nk->ni", Gm + Gm.transpose(0, 2, 1), n3)
            Bw[g, zid[m], 0] = (trac * t3).sum(1)
            Bw[g, zid[m], 1] = trac[:, 2]
            Bw[g, zid[m], 2] = pr[m]
    spec = np.stack([W_SPEC_V @ Bv[:, :, c] for c in range(3)], -1)
    curve = np.stack([W_CURVE_V @ Bv[:, :, c] for c in range(3)], -1)
    return path, header(path)["time"], sums, spec.astype(np.float32), curve.astype(np.float32), Bw.astype(np.float32)


# ------------------------------------------------------------------------------- rotation into the line frame
def frame_q(fr):
    """(n, 4) [t, n] -> Q (n, 3, 3), rows s, n, z."""
    Q = np.zeros((len(fr), 3, 3))
    Q[:, 0, :2], Q[:, 1, :2], Q[:, 2, 2] = fr[:, :2], fr[:, 2:], 1
    return Q


def sym_full(comps, combos, order):
    """(npts, len(combos)) symmetric components -> full tensor (npts, 3, ..., 3)."""
    full = np.zeros((comps.shape[0],) + (3,) * order)
    for c, ijk in enumerate(combos):
        for perm in set(itertools.permutations(ijk)):
            full[(slice(None),) + perm] = comps[:, c]
    return full


def rot2(Q, comps):
    """Symmetric second-order tensor (npts, 6 in C2 order) -> rotated, (npts, 6) [ss, sn, sz, nn, nz, zz]."""
    R = np.einsum("pai,pij,pbj->pab", Q, sym_full(comps, C2, 2), Q)
    return np.stack([R[:, a, b] for a, b in C2], 1)


# ------------------------------------------------------------------------------------------------- run
TOT = np.zeros((NQ, NG))
times, per_field, spec_s, curve_s, wall_s = [], [], [], [], []
ctx = multiprocessing.get_context("fork")
nw = max(1, min(args.workers, len(files)))
log(f"accumulating {len(files)} fields with {nw} workers ({NQ} quantities per 2D point)")
with ctx.Pool(nw) as pool:
    for path, t, sums, spec, curve, bw in pool.imap(accumulate, files):
        TOT += sums
        times.append(t)
        # per field: U, p and R (about the field's own mean) on the lines, for the confidence intervals
        cnt = CNT
        m1 = sums[[IDX["u0"], IDX["u1"], IDX["u2"]]] / cnt
        r2 = np.stack([sums[IDX[f"uu{a}{b}"]] / cnt - m1[a] * m1[b] for a, b in C2])
        v = W_LINES @ np.c_[m1.T, sums[IDX["p"]] / cnt, r2.T]
        Q = frame_q(frames)
        per_field.append(np.c_[np.einsum("pai,pi->pa", Q, v[:, :3]), v[:, 3], rot2(Q, v[:, 4:])].astype(np.float32))
        spec_s.append(spec)
        curve_s.append(curve)
        wall_s.append(bw)
        log(f"{os.path.basename(path)} t = {t:.4f} done ({len(times)}/{len(files)})")

# -------------------------------------------------------------------------------------- central moments
M = TOT / (CNT * len(files))
Um = M[[IDX["u0"], IDX["u1"], IDX["u2"]]]
Pm = M[IDX["p"]]


def key(*ijk):
    return tuple(sorted(ijk))


raw2 = {c: M[IDX[f"uu{c[0]}{c[1]}"]] for c in C2}
raw3 = {c: M[IDX["uuu" + "".join(map(str, c))]] for c in C3}
raw4 = {c: M[IDX["uuuu" + "".join(map(str, c))]] for c in C4}
R = {c: raw2[c] - Um[c[0]] * Um[c[1]] for c in C2}
T3 = {}
for i, j, k in C3:
    T3[(i, j, k)] = (
        raw3[(i, j, k)]
        - Um[i] * raw2[key(j, k)]
        - Um[j] * raw2[key(i, k)]
        - Um[k] * raw2[key(i, j)]
        + 2 * Um[i] * Um[j] * Um[k]
    )
T4 = {}
for i, j, k, l in C4:
    U_ = Um
    T4[(i, j, k, l)] = (
        raw4[(i, j, k, l)]
        - U_[i] * raw3[key(j, k, l)]
        - U_[j] * raw3[key(i, k, l)]
        - U_[k] * raw3[key(i, j, l)]
        - U_[l] * raw3[key(i, j, k)]
        + U_[i] * U_[j] * raw2[key(k, l)]
        + U_[i] * U_[k] * raw2[key(j, l)]
        + U_[i] * U_[l] * raw2[key(j, k)]
        + U_[j] * U_[k] * raw2[key(i, l)]
        + U_[j] * U_[l] * raw2[key(i, k)]
        + U_[k] * U_[l] * raw2[key(i, j)]
        - 3 * U_[i] * U_[j] * U_[k] * U_[l]
    )
pp = M[IDX["pp"]] - Pm**2
pu = np.stack([M[IDX[f"pu{i}"]] - Pm * Um[i] for i in range(3)])
dU = np.array([[M[IDX[f"G{i}{k}"]] for k in range(3)] for i in range(3)])  # (i, k, NG)
dP = np.stack([M[IDX[f"dp{k}"]] for k in range(3)])
eps = {(a, b): 2 * NU * (M[IDX[f"GG{a}{b}"]] - (dU[a] * dU[b]).sum(0)) for a, b in C2}
pg = np.array([[M[IDX[f"pG{i}{j}"]] - Pm * dU[i, j] for j in range(3)] for i in range(3)])
udp = np.array([[M[IDX[f"udp{i}{j}"]] - Um[i] * dP[j] for j in range(3)] for i in range(3)])
phi = {(a, b): pg[a, b] + pg[b, a] for a, b in C2}
Pi = {(a, b): -(udp[a, b] + udp[b, a]) for a, b in C2}
Rf = np.zeros((3, 3, NG))
for (a, b), v in R.items():
    Rf[a, b] = Rf[b, a] = v
prod = {(a, b): -((Rf[a] * dU[b]).sum(0) + (Rf[b] * dU[a]).sum(0)) for a, b in C2}

# terms with derivatives of averaged quantities, from the accumulated products (exact derivatives of the element
# polynomials, no differentiation of averaged fields; continuity div u = 0 used)
uG = np.array([[[M[IDX[f"uG{i}{j}{k}"]] for k in range(3)] for j in range(3)] for i in range(3)])  # <u_i du_j/dx_k>
dR = uG + uG.transpose(1, 0, 2, 3) - (Um[:, None, None] * dU[None] + Um[None, :, None] * dU[:, None])  # dR_ij/dx_k
uA = np.array([[M[IDX[f"uA{j}{i}"]] for j in range(3)] for i in range(3)])  # [i, j] = <u_j A_i>, A = u . grad u
A = np.stack([M[IDX[f"A{i}"]] for i in range(3)])
uL = np.array([[M[IDX[f"uL{i}{j}"]] for j in range(3)] for i in range(3)])  # <u_i lap u_j>
Lm = np.stack([M[IDX[f"L{i}"]] for i in range(3)])
raw2f = np.zeros((3, 3, NG))
for (a, b), v in raw2.items():
    raw2f[a, b] = raw2f[b, a] = v
conv, visc, ttr = {}, {}, {}
for i, j in C2:
    conv[(i, j)] = -(Um * dR[i, j]).sum(0)
    # nu lap R_ij = nu (<u_i' lap u_j'> + <u_j' lap u_i'>) + eps_ij
    visc[(i, j)] = NU * (uL[i, j] - Um[i] * Lm[j] + uL[j, i] - Um[j] * Lm[i]) + eps[(i, j)]
    # d<u_i' u_j' u_k'>/dx_k from the raw moments
    div3 = (
        uA[i, j]
        + uA[j, i]
        - ((dU[i] * raw2f[j]).sum(0) + Um[i] * A[j])
        - ((dU[j] * raw2f[i]).sum(0) + Um[j] * A[i])
        - (Um * (uG[i, j] + uG[j, i])).sum(0)
        + 2 * (Um * (Um[j] * dU[i] + Um[i] * dU[j])).sum(0)
    )
    ttr[(i, j)] = -div3

# ------------------------------------------------------------------------ KTH int_fld (NekStatsPy input)
# The 66 statistics and 68 derivatives of the KTH NACA0012 post-processing (pstat2D, read_data_profiles of
# NekStatsPy): F = raw moments <..> of the 2D (z and field) averages, D = derivatives of the averaged fields,
# computed element by element as comp_derivat of the KTH naca_wing.usr (here: averaged at shared nodes), then
# interpolated onto the points of the interpolation mesh. Statistics that NekStatsPy does not read and that are
# not accumulated here are zero: F27 <ppp>, F38 <pppp>, F48-60 single gradient products, F61-63 forcing and
# F64-66 RT-filter dissipation (they only enter its DRT term).
KTH_F = {
    1: "u0",
    2: "u1",
    3: "u2",
    4: "p",
    5: "uu00",
    6: "uu11",
    7: "uu22",
    8: "pp",
    9: "uu01",
    10: "uu12",
    11: "uu02",
    12: "pu0",
    13: "pu1",
    14: "pu2",
    **{15 + 3 * i + j: f"pG{i}{j}" for i in range(3) for j in range(3)},
    24: "uuu000",
    25: "uuu111",
    26: "uuu222",
    28: "uuu001",
    29: "uuu002",
    30: "uuu011",
    31: "uuu112",
    32: "uuu022",
    33: "uuu122",
    34: "uuu012",
    35: "uuuu0000",
    36: "uuuu1111",
    37: "uuuu2222",
    39: "uuuu0001",
    40: "uuuu0011",
    41: "uuuu0111",
    42: "GG00",
    43: "GG11",
    44: "GG22",
    45: "GG01",
    46: "GG02",
    47: "GG12",
}
# first derivatives: F index -> (D index of d/dx, D index of d/dy)
KTH_D1 = {
    1: (1, 2),
    2: (3, 4),
    3: (5, 6),
    4: (7, 8),
    5: (9, 10),
    6: (11, 12),
    7: (13, 14),
    8: (15, 16),
    9: (17, 18),
    10: (19, 20),
    11: (21, 22),
    24: (23, 24),
    25: (25, 26),
    26: (27, 28),
    27: (29, 30),
    28: (31, 32),
    29: (33, 34),
    30: (35, 36),
    31: (37, 38),
    32: (39, 40),
    33: (41, 42),
    34: (43, 44),
    12: (63, 64),
    13: (65, 66),
    14: (67, 68),
}
# second derivatives: D index -> (D index of the result, direction 0 = x, 1 = y)
KTH_D2 = {
    1: (45, 0),
    2: (46, 1),
    3: (47, 0),
    4: (48, 1),
    5: (49, 0),
    6: (50, 1),
    9: (51, 0),
    10: (52, 1),
    11: (53, 0),
    12: (54, 1),
    13: (55, 0),
    14: (56, 1),
    17: (57, 0),
    18: (58, 1),
    21: (59, 0),
    22: (60, 1),
    19: (61, 0),
    20: (62, 1),
}
DFIELDS = None  # (nf, NG) fields differentiated by the workers (fork)


def grad_chunk(e0):
    """Sums over the 2D points of d/dx and d/dy of DFIELDS on the elements e0 .. e0 + chunk."""
    e1 = min(NEL, e0 + args.chunk_elements)
    X, _, _ = open_field(files[0])
    Jinv = jacobian_inv(np.asarray(X[e0:e1], dtype=float))
    lab = LAB[e0 * NPT : e1 * NPT]
    out = np.zeros((len(DFIELDS), 2, NG))
    for q, f in enumerate(DFIELDS):
        g = grad(f[lab].reshape(e1 - e0, NPT), Jinv)
        for d in range(2):
            out[q, d] += np.bincount(lab, g[:, d], NG)
    return out


def grad_2d(fields):
    """d/dx, d/dy of 2D fields (nf, NG): element derivatives, averaged over the element copies of each 2D point."""
    global DFIELDS
    DFIELDS = fields  # before the fork: the workers inherit it
    tot = np.zeros((len(fields), 2, NG))
    with ctx.Pool(max(1, args.workers)) as pool:
        for s in pool.imap_unordered(grad_chunk, range(0, NEL, args.chunk_elements)):
            tot += s
    return tot / CNT


def write_int_fld(mesh_file):
    import scipy.io as sio

    mesh = sio.loadmat(mesh_file, squeeze_me=True)
    xq, yq = np.asarray(mesh["x_pts"], float), np.asarray(mesh["y_pts"], float)
    ln = len(np.flatnonzero(xq == -1))
    F = np.zeros((67, NG))
    for k, name in KTH_F.items():
        F[k] = M[IDX[name]]
    D = np.zeros((69, NG))
    src = sorted(KTH_D1)
    g = grad_2d(F[src])
    for q, k in enumerate(src):
        D[KTH_D1[k][0]], D[KTH_D1[k][1]] = g[q, 0], g[q, 1]
    log(f"int_fld: first derivatives of {len(src)} fields")
    src = sorted(KTH_D2)
    g = grad_2d(D[src])
    for q, k in enumerate(src):
        D[KTH_D2[k][0]] = g[q, KTH_D2[k][1]]
    log(f"int_fld: second derivatives of {len(src)} fields")
    # wall points (the first of every profile but the yn column at x = -1): linear between the two nearest wall
    # nodes of the same surface; all other points linear on the Delaunay triangulation
    wall_q = np.arange(0, len(xq), ln)
    wall_q = wall_q[xq[wall_q] != -1]
    Wq, ok = interp_matrix(np.c_[xq, yq])
    Wq = Wq.tolil()
    for r in wall_q:
        side = 1 if yq[r] >= 0 else -1
        k = np.flatnonzero((np.sign(SY) == side) | (np.abs(SY) < 1e-9))
        d = np.hypot(SX[k] - xq[r], SY[k] - yq[r])
        a, b = k[np.argsort(d)[:2]]
        da, db = np.hypot(SX[a] - xq[r], SY[a] - yq[r]), np.hypot(SX[b] - xq[r], SY[b] - yq[r])
        Wq.rows[r], Wq.data[r] = [], []
        Wq[r, WALLG[a]] = db / (da + db) if da + db > 0 else 1.0
        Wq[r, WALLG[b]] = da / (da + db) if da + db > 0 else 0.0
        ok[r] = True
    Wq = Wq.tocsr()
    bad = np.flatnonzero(~ok)
    if len(bad):  # outside the triangulation: nearest 2D point
        Wq = Wq.tolil()
        for r, c in zip(bad, tree_g.query(np.c_[xq[bad], yq[bad]])[1]):
            Wq.rows[r], Wq.data[r] = [c], [1.0]
        Wq = Wq.tocsr()
    Fq, Dq = Wq @ F[1:].T, Wq @ D[1:].T
    os.makedirs(os.path.join(args.run, "ZSTAT"), exist_ok=True)
    path = os.path.join(args.run, "ZSTAT", "int_fld")
    # header as read by read_header of NekStatsPy (read_data_profiles.m); records 8 bytes apart
    hdr = f"NekRS {os.path.normpath(args.run)}: bl_stats.py --int-mesh, {len(files)} fields".encode()
    with open(path, "wb") as f:
        np.array([len(hdr)], "<i4").tofile(f)
        f.write(hdr)
        np.array([0.0, 1 / NU, 0.0, 0.0, 0.0], "<f8").tofile(f)  # dum5, Rer, Domain
        np.array([NEL, 1, 1, LX - 1, LX - 1, LX - 1, 66, 68], "<i4").tofile(f)  # nel, Poly, nstat, nderiv
        np.array([min(times), max(times), max(times) - min(times), 0.0], "<f8").tofile(f)  # times, timee, atime, DT
        np.array([len(files)], "<i4").tofile(f)
        np.array([0.0], "<f8").tofile(f)  # Tint
        np.array([len(xq)], "<i4").tofile(f)
        np.array([0.0], "<f8").tofile(f)  # dum6
        for i, col in enumerate(np.concatenate([Fq, Dq], axis=1).T):
            if i:
                f.write(b"\0" * 8)
            col.astype("<f8").tofile(f)
    log(f"{path} written: {len(xq)} points of {mesh_file} ({len(bad)} outside the mesh, nearest 2D point)")


if args.int_mesh:
    write_int_fld(args.int_mesh)

# ----------------------------------------------------------------------------------------- line values
log("interpolation onto the lines")
Q = frame_q(frames)


def on_lines(d):
    return W_LINES @ np.stack([d[c] for c in sorted(d)], 1)


Ul = W_LINES @ Um.T
lines_out = {
    "U": np.einsum("pai,pi->pa", Q, Ul),
    "P": W_LINES @ Pm,
    "R": rot2(Q, on_lines(R)),
    "pp": W_LINES @ pp,
    "pu": np.einsum("pai,pi->pa", Q, W_LINES @ pu.T),
    "dU": np.einsum("pai,pij,pbj->pab", Q, (W_LINES @ dU.reshape(9, NG).T).reshape(-1, 3, 3), Q).reshape(-1, 9),
    "dP": np.einsum("pai,pi->pa", Q, W_LINES @ dP.T),
}
t3 = sym_full(on_lines(T3), C3, 3)
t4 = sym_full(on_lines(T4), C4, 4)
lines_out["m3"] = np.einsum("pai,paj,pak,pijk->pa", Q, Q, Q, t3)  # <u_a'^3>, a = s, n, z
lines_out["m4"] = np.einsum("pai,paj,pak,pal,pijkl->pa", Q, Q, Q, Q, t4)
for name, d in (
    ("prod", prod),
    ("eps", eps),
    ("phi", phi),
    ("pdiff", {c: Pi[c] - phi[c] for c in C2}),
    ("ttrans", ttr),
    ("vdiff", visc),
    ("conv", conv),
):
    lines_out["budget_" + name] = rot2(Q, on_lines(d))
for k_, v in lines_out.items():
    v[~valid] = np.nan

# free-stream total pressure (upstream far field) and the stagnation pressure of the 2D field
up = GX < -1.5
p0 = float(np.mean(Pm[up] + 0.5 * (Um[:, up] ** 2).sum(0)))
p_stag = float(Pm[WALLG][SX < 0.01].max())

meta = {
    "run": os.path.normpath(args.run),
    "skip": args.skip,
    "files": [os.path.basename(f) for f in files],
    "times": times,
    "nu": NU,
    "Lz": Lz,
    "nz": NZ,
    "N": LX - 1,
    "elements": NEL,
    "points_2d": NG,
    "p0_freestream": p0,
    "p_stag": p_stag,
    "line_components": "s (downstream tangent), n (wall normal into the fluid), z",
    "tensor_components": ["ss", "sn", "sz", "nn", "nz", "zz"],
    "wake_frame": "x, y, z",
    "budget_sign": "terms of dR_ij/dt = conv + prod + phi + pdiff + ttrans + vdiff - eps (eps positive)",
    "args": vars(args),
}
lw = np.array([l[0] == "wall" for l in lines])
np.savez_compressed(
    out_file,
    meta=json.dumps(meta),
    # 2D fields
    gx=GX.astype(np.float32),
    gy=GY.astype(np.float32),
    gU=Um.T.astype(np.float32),
    gP=Pm.astype(np.float32),
    gR=np.stack([R[c] for c in C2], 1).astype(np.float32),
    # lines
    line_kind=np.array([l[0] for l in lines]),
    line_x=np.array([l[1] for l in lines]),
    line_side=np.array([l[2] for l in lines]),
    line_origin=np.array([l[3] for l in lines]),
    line_t=np.array([l[4] for l in lines]),
    line_n=np.array([l[5] for l in lines]),
    line_of=line_of,
    line_pts=pts,
    nvals=nvals,
    wake_y=yw,
    **{"L_" + k_: v for k_, v in lines_out.items()},
    per_field=np.array(per_field),  # (fields, points, [U_s, U_n, W, P, R ss sn sz nn nz zz])
    # z-lines
    zs=ZS,
    spec_pts=spec_pts,
    spec=np.array(spec_s),  # (fields, spec points, z, [u, v, w] in x, y, z)
    curve_pts=curve_pts,
    curve_t=curve_t,
    curve_x=xc,
    curve_n=np.array(args.curve_n),
    curve_valid=cvalid,
    curve=np.array(curve_s),  # (fields, [side, n, x], z, [u, v, w])
    wall_x=SX,
    wall_y=SY,
    wall_n=SN,
    wall_t=ST,
    wall_dn1=dn1,
    wall_ds=ds_wall,
    wall=np.array(wall_s),  # (fields, stations, z, [tau_s, tau_z, p])
)
log(f"{out_file} written")
