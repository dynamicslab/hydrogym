"""Actuation power of wall jets of m-AIA's boundary condition 2007 on general walls (flat faces, airfoils, spheres).

``hydrogym.maia.power.SlotJetPower`` covers slot jets on circular cylinders. This module covers the other jet
geometries of BC 2007 (``bc20007`` in ``src/LB/lbbndcnddxqy.cpp``): flat plate (``lbObjectClass`` 1, e.g. the open
cavity and the NACA 0012 airfoil), cube (2), sphere (3) and the circular-body rule (0) on flat faces (square
cylinder, cavity cases with ``lbJetRadius`` 0).

**Jet geometry (as the solver does it).** ``Bc2007Jets`` reproduces the tagging of BC 2007: a boundary cell
belongs to jet ``j`` if its centre lies in the jet's bounding box (``lbJetCenter`` +- ``lbJetRanges`` / 2, cut by
``lbJetLimits``) and its distance from the jet axis is within the half range (rule per object class: distance from
the jet centre (class 1), across the slot axis (class 2), as arc length along the slot axis plus an azimuthal range
(class 3), across the jet direction (class 0)); its wall velocity is ``command_j * factor`` with the parabolic factor
``1 - d^2 / h^2`` times the jet direction (``lbJetAngles``, ``lbJetNormals`` or the sphere radial). The 2D
last-box-wins tagging of the solver is kept. The factor is evaluated at the centre of the fluid cell next to the
wall (the cell whose links carry the wall velocity), not at the wall itself:

* flat walls on cell faces (cavity, square cylinder, cube): the jet consists exactly of the fluid cells adjacent to
  the wall, one wall element per cell (area ``dX^(nDim-1)``), factor at the cell centre (``flat_wall_jets``);
* curved walls (airfoil from the case's geometry file, sphere) and flat walls off the cell faces: the moving-wall
  bounce-back is modelled link by link (``link_jets``): which boundary cell writes each cut link (the fluid cell
  for ``q <= 1/2``, the solid cell across it otherwise, as the solver's redundant-cut correction does), its moving-
  wall source, and the flow-dependent part of the interpolated bounce-back, which ``WallJetPower`` evaluates
  from the probed cell values. On the NACA 0012 (``c/128``, jet centres inside the airfoil) these terms are 12-36 %
  of the jet mass flux; with them the model reproduces the solver's mass flux (control volume) within 4 % (within
  1.2 % on the cavity walls off the cell faces, where they are up to 29 %).
* ``sphere_jets`` and ``curve_jets`` are continuous quadratures of the jet profile on the wall (cross-checks).

**Power.** ``P = int v_n (p - p_ref) dA + int v_n rho |u_w|^2 / 2 dA`` point-wise per wall element (Bewley, Moin &
Temam 2001; ``hydrogym.power.transpiration_forms``) with ``v_n = u_w . n`` (``n`` into the fluid, ``v_n > 0``
blowing; with the interpolation terms of ``link_jets`` the actual mass flux per element divided by ``rho dA``).
The wall pressure and density are extrapolated to the wall from the cell-centre values at ``wall_distances``
(default 1.5-4.5 cells) along the wall normal (quadratic least squares; m-AIA returns probe values by nearest
neighbour, so the model probes the cell centres of the finest level itself and interpolates multilinearly); the
relative difference between the quadratic and the linear extrapolation is kept as ``last_extrapolation_spread``.
``p_ref`` is the area-mean wall pressure over all jets of the environment (one actuator
group). Where the jets are not zero-net-mass-flux (independent actions), the signed power then does not depend on
the pressure gauge but leaves out the work ``p_ref Q`` of supplying the net flow ``Q`` (as in the Firedrake cavity
and cylinder environments).

Normalisation: ``C_P = 2 P / (U^3 L_proj)`` with ``U = Ma / sqrt(3)`` and ``L_proj`` the projection length (2D, cells)
or area (3D, cells^2) of the environment's drag coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from hydrogym.power import transpiration_forms

_BIG = np.finfo(float).max


# ----------------------------------------------------------------------------------------------------------- BC 2007
@dataclass
class Bc2007Jets:
    """Jet definition of m-AIA's BC 2007, from the solver properties (geometry units).

    ``centers`` are the jet centres after m-AIA's shift (object class 0: ``lbJetCenter + lbJetRadius (cos, sin)``),
    ``limits`` the box offsets per jet ``[-x, -y(, -z), +x, +y(, +z)]`` with non-positive entries = no limit.
    """

    n_dim: int
    object_class: int
    centers: np.ndarray  # (nj, nDim)
    angles: np.ndarray  # (nj,)
    ranges: np.ndarray  # (nj,)
    limits: np.ndarray  # (nj, 2 nDim), +inf where unlimited
    normals: Optional[np.ndarray] = None  # (nj, nDim), class 2
    slot_axes: Optional[np.ndarray] = None  # (nj,), classes 2 and 3
    sphere_center: Optional[np.ndarray] = None  # (nDim,), class 3
    sphere_radius: float = 0.0  # class 3: lbJetRadius[0]

    @property
    def n_jets(self) -> int:
        return len(self.centers)

    @classmethod
    def from_properties(cls, props: dict) -> "Bc2007Jets":
        """From the property dict of the case (``properties_run.toml``)."""
        n_dim = int(props["nDim"])
        nj = int(props["lbNoJets"])
        oc = int(props.get("lbObjectClass", 0))
        centers = np.asarray(props["lbJetCenter"], dtype=float).reshape(nj, n_dim).copy()
        angles = np.asarray(props.get("lbJetAngles", [0.0] * nj), dtype=float).reshape(nj)
        ranges = np.asarray(props["lbJetRanges"], dtype=float).reshape(nj)
        radii = np.asarray(props.get("lbJetRadius", [0.0] * nj), dtype=float).reshape(-1)
        radii = np.broadcast_to(radii, (nj,)) if radii.size in (1, nj) else radii[:nj]
        if "lbJetLimits" in props:
            lim = np.asarray(props["lbJetLimits"], dtype=float).reshape(nj, 2 * n_dim)
            lim = np.where(lim > 0.0, lim, np.inf)
        else:
            lim = np.full((nj, 2 * n_dim), np.inf)
        if oc == 0:
            centers[:, 0] += radii * np.cos(angles)
            centers[:, 1] += radii * np.sin(angles)
        normals = slot = sc = None
        sr = 0.0
        if oc == 2:
            normals = np.asarray(props["lbJetNormals"], dtype=float).reshape(nj, n_dim)
            slot = np.asarray(props.get("lbJetSlotAxis", [0] * nj), dtype=int).reshape(nj)
        if oc == 3:
            sc = np.asarray(props["lbJetCenterSph"], dtype=float).reshape(n_dim)
            slot = np.asarray(props.get("lbJetSlotAxis", [0] * nj), dtype=int).reshape(nj)
            sr = float(radii[0])
        return cls(n_dim, oc, centers, angles, ranges, lim, normals, slot, sc, sr)

    def boxes(self):
        """Bounding boxes (lo, hi), each (nj, nDim), as in the solver (open intervals)."""
        d, oc = self.n_dim, self.object_class
        lo = np.empty((self.n_jets, d))
        hi = np.empty((self.n_jets, d))
        for j in range(self.n_jets):
            c, L = self.centers[j], self.limits[j]
            ext = self.sphere_radius * self.ranges[j] if oc == 3 else self.ranges[j]
            if oc == 3 and self.ranges[j] > 0.8:  # large sphere jets: the whole sphere plus a margin
                margin = max(L[0], L[d]) + 1.0
                lo[j] = self.sphere_center - self.sphere_radius - margin
                hi[j] = self.sphere_center + self.sphere_radius + margin
                continue
            for k in range(min(d, 2)):
                lo[j, k] = max(c[k] - 0.5 * ext, c[k] - L[k])
                hi[j, k] = min(c[k] + 0.5 * ext, c[k] + L[k + d])
            if d == 3:
                lo[j, 2] = c[2] - L[2]
                hi[j, 2] = c[2] + L[5]
                if oc == 3:
                    lo[j, 2] = min(lo[j, 2], c[2] - 0.5 * ext)
                    hi[j, 2] = max(hi[j, 2], c[2] + 0.5 * ext)
        return lo, hi

    def tag(self, x: np.ndarray):
        """Jet index (-1: none) and multiplication factor vector (wall velocity per unit command) of cells at ``x``.

        ``x``: cell centres (n, nDim), geometry units. Reproduces the kernel of ``bc20007`` (first call).
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.n_dim)
        n, d, oc = len(x), self.n_dim, self.object_class
        tag = np.full(n, -1, dtype=int)
        fac = np.zeros((n, d))
        lo, hi = self.boxes()
        for j in range(self.n_jets):
            inbox = np.all((x > lo[j]) & (x < hi[j]), axis=1)
            dx = x - self.centers[j]
            sj, cj = np.sin(self.angles[j]), np.cos(self.angles[j])
            inside = np.ones(n, dtype=bool)
            if d == 2:
                perp2 = (dx[:, 0] ** 2 + dx[:, 1] ** 2) if oc == 1 else (dx[:, 0] * sj - dx[:, 1] * cj) ** 2
                half2 = (0.5 * self.ranges[j]) ** 2
            elif oc == 3:
                ax = int(self.slot_axes[j])
                R = self.sphere_radius
                vc = x - self.sphere_center
                th_c = np.arccos(np.clip(vc[:, ax] / R, -1.0, 1.0))
                th_j = np.arccos(np.clip((self.centers[j, ax] - self.sphere_center[ax]) / R, -1.0, 1.0))
                perp2 = (R * (th_c - th_j)) ** 2
                half2 = min(self.limits[j, ax], self.limits[j, ax + d]) ** 2
                vj = self.centers[j] - self.sphere_center
                vc0, vj0 = vc.copy(), vj.copy()
                vc0[:, ax] = 0.0
                vj0[ax] = 0.0
                mc, mj = np.linalg.norm(vc0, axis=1), np.linalg.norm(vj0)
                ok = (mc > 1e-12) & (mj > 1e-12)
                cos_t = np.where(ok, (vc0 @ vj0) / np.where(ok, mc * mj, 1.0), -1.0)
                inside = ~((perp2 <= half2) & (cos_t < np.cos(0.5 * self.ranges[j])))
            elif oc == 2:
                ax = int(self.slot_axes[j])
                perp2 = dx[:, ax] ** 2
                half2 = min(self.limits[j, ax], self.limits[j, ax + d]) ** 2
            elif oc == 1:
                perp2 = dx[:, 0] ** 2 + dx[:, 1] ** 2
                half2 = (0.5 * self.ranges[j]) ** 2
            else:
                perp2 = (dx[:, 0] * sj - dx[:, 1] * cj) ** 2
                half2 = (0.5 * self.ranges[j]) ** 2
            prof = inbox & (perp2 <= half2) & inside
            f = 1.0 - perp2 / half2
            if oc == 2 and d == 3:
                direction = np.broadcast_to(self.normals[j], (n, d))
            elif oc == 3 and d == 3:
                vc = x - self.sphere_center
                r = np.linalg.norm(vc, axis=1, keepdims=True)
                direction = np.where(r > 1e-10, vc / np.where(r > 1e-10, r, 1.0), 0.0)
            else:
                direction = np.zeros((n, d))
                direction[:, 0], direction[:, 1] = cj, sj
            if d == 2:
                tag[inbox] = j  # 2D: the box decides the tag, the profile test the factor (solver order)
            else:
                tag[prof] = j
            fac[prof] = f[prof, None] * direction[prof]
        return tag, fac


# ----------------------------------------------------------------------------------------------------- wall elements
@dataclass
class WallElements:
    """Discretised jet walls: one entry per wall element.

    points: wall points (n, nDim), geometry units; normals: unit direction of the element's flux vector (into the
    fluid); weights: element area in lattice units (cells^(nDim-1)); factors: wall velocity per unit command (n,
    nDim); jet: jet index (n,); inside: callable, True for points inside the solid (checks the pressure stencils);
    sample_normals: wall normals along which the pressure is sampled (default: normals); links: the cut lattice links
    of the elements (``link_jets``), whose interpolation terms make the mass flux flow-dependent.
    """

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    factors: np.ndarray
    jet: np.ndarray
    inside: Optional[Callable[[np.ndarray], np.ndarray]] = None
    sample_normals: Optional[np.ndarray] = None
    links: Optional[dict] = None

    def __len__(self):
        return len(self.weights)

    @property
    def sampling_normals(self) -> np.ndarray:
        return self.normals if self.sample_normals is None else self.sample_normals

    def subset(self, mask) -> "WallElements":
        """The elements selected by ``mask`` (with their links)."""
        idx = np.flatnonzero(mask)
        links = None
        if self.links is not None:
            remap = np.full(len(self), -1)
            remap[idx] = np.arange(len(idx))
            keep = remap[self.links["elem"]] >= 0
            links = {k: v[keep] for k, v in self.links.items()}
            links["elem"] = remap[links["elem"]]
        return WallElements(
            self.points[idx],
            self.normals[idx],
            self.weights[idx],
            self.factors[idx],
            self.jet[idx],
            self.inside,
            self.sampling_normals[idx],
            links,
        )


def _assign(jets: Bc2007Jets, points, normals, weights, offsets, inside=None) -> WallElements:
    """Tag the cells at ``points + offsets * normals`` and keep the elements with a jet velocity."""
    tag, fac = jets.tag(points + offsets[:, None] * normals)
    keep = (tag >= 0) & np.any(fac != 0.0, axis=1)
    return WallElements(points[keep], normals[keep], weights[keep], fac[keep], tag[keep], inside)


def _cell_centres(lo: float, hi: float, origin: float, dX: float) -> np.ndarray:
    """Finest-level cell centres in the open interval (lo, hi)."""
    i0 = int(np.ceil((lo - origin) / dX - 0.5 - 1e-9))
    i1 = int(np.floor((hi - origin) / dX - 0.5 + 1e-9))
    c = origin + (np.arange(i0, i1 + 1) + 0.5) * dX
    return c[(c > lo) & (c < hi)]


def flat_wall_jets(
    jets: Bc2007Jets,
    dX: float,
    lattice_origin: Sequence[float],
    span: Optional[Sequence[float]] = None,
    max_cells_per_axis: int = 64,
    inside: Optional[Callable] = None,
) -> WallElements:
    """Jets on flat walls aligned with the lattice axes, exactly as the solver discretises them.

    The wall of jet ``j`` is the plane through its centre normal to its blowing direction (object classes 0 and 1:
    ``lbJetAngles``; class 2: ``lbJetNormals``), which must be a lattice axis. If the plane lies on cell faces, its
    jet cells are the fluid cells adjacent to it whose centres pass the BC 2007 test (``Bc2007Jets.tag``); each
    is one wall element of area ``dX^(nDim-1)`` at the cell's wall foot point, with the factor of the cell centre.
    Otherwise (2D) the jet is modelled link by link (``link_jets`` with the half plane as the solid).

    Args:
        span: (lo, hi) of the domain along z (3D), used where the jet box is not limited in z.
        max_cells_per_axis: in 3D, along a wall direction in which the jet profile is uniform (no box limit inside the
            domain other than the extent), every k-th cell is used (with k-fold weight) so that at most this many
            remain; the profile direction is always resolved cell by cell.
    """
    d = jets.n_dim
    origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[:d]
    lo_b, hi_b = jets.boxes()
    pts, nrm, wts, offs, extra = [], [], [], [], []
    for j in range(jets.n_jets):
        if jets.object_class == 2:
            direc = np.asarray(jets.normals[j], dtype=float)
        elif jets.object_class == 3:
            raise ValueError("flat_wall_jets: object class 3 (sphere) has curved walls, use sphere_jets")
        else:
            direc = np.zeros(d)
            direc[0], direc[1] = np.cos(jets.angles[j]), np.sin(jets.angles[j])
        ax = int(np.argmax(np.abs(direc)))
        if abs(abs(direc[ax]) - 1.0) > 1e-4:
            raise ValueError(f"flat_wall_jets: jet {j} does not blow along a lattice axis ({direc})")
        sgn = np.sign(direc[ax])
        wall = jets.centers[j, ax]
        # first fluid cell centre on the blowing side of the wall
        k = (wall - origin[ax]) / dX - 0.5
        kf = np.floor(k + 1e-9) + 1 if sgn > 0 else np.ceil(k - 1e-9) - 1
        off = abs(origin[ax] + (kf + 0.5) * dX - wall)
        if abs(off - 0.5 * dX) > 0.5e-3 * dX:  # (grid header coordinates carry ~1e-6 cell round-off)
            # wall off the cell faces: interpolated bounce-back, and for q > 1/2 the solid cells write: link model
            if d != 2:
                raise ValueError("flat_wall_jets: walls off the cell faces are supported in 2D only")
            extra.append(_half_space_link_jet(jets, j, ax, sgn, wall, dX, origin))
            continue
        tang = [t for t in range(d) if t != ax]
        axes_c = []
        for t in tang:
            lo, hi = lo_b[j, t], hi_b[j, t]
            if t == 2 and span is not None:
                lo, hi = max(lo, span[0]), min(hi, span[1])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"flat_wall_jets: jet {j} is not bounded along axis {t}; give the span")
            axes_c.append(_cell_centres(lo, hi, origin[t], dX))
        # profile direction: the one with the parabolic variation; the others may be thinned
        stride = [1] * len(tang)
        if d == 3:
            for q, t in enumerate(tang):
                prof_axis = (jets.object_class == 2 and int(jets.slot_axes[j]) == t) or (
                    jets.object_class in (0, 1) and t != 2
                )
                if not prof_axis and len(axes_c[q]) > max_cells_per_axis:
                    stride[q] = int(np.ceil(len(axes_c[q]) / max_cells_per_axis))
        sel, wq = [], []
        for q, c in enumerate(axes_c):
            s = stride[q]
            keep_idx = np.arange(s // 2, len(c), s) if s > 1 else np.arange(len(c))
            sel.append(c[keep_idx])
            # each kept cell stands for the cells of its block (exact count at the ends)
            wq.append((np.minimum(keep_idx - s // 2 + s, len(c)) - (keep_idx - s // 2)).astype(float))
        grids = np.meshgrid(*sel, indexing="ij")
        W = np.prod(np.stack([g.ravel() for g in np.meshgrid(*wq, indexing="ij")]), axis=0)
        P = np.zeros((grids[0].size, d))
        for q, t in enumerate(tang):
            P[:, t] = grids[q].ravel()
        P[:, ax] = wall
        n = np.zeros_like(P)
        n[:, ax] = sgn
        pts.append(P)
        nrm.append(n)
        wts.append(W)
        offs.append(np.full(len(P), off))
    parts = list(extra)
    if pts:
        P, N, W, off = (np.concatenate(a) for a in (pts, nrm, wts, offs))
        # one element per cell: a cell inside the boxes of two jets on the same wall is counted once (the solver
        # gives it one tag)
        cell = np.round((P + off[:, None] * N - origin) / dX - 0.5).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        first = np.sort(first)
        parts.insert(0, _assign(jets, P[first], N[first], W[first], off[first], inside))
    return parts[0] if len(parts) == 1 else concat_elements(parts, inside)


def concat_elements(parts: Sequence[WallElements], inside=None) -> WallElements:
    """Concatenation of element sets."""
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
    links = None
    if any(p.links is not None for p in parts):
        ls, off = [], 0
        for p in parts:
            if p.links is not None:
                lk = dict(p.links)
                lk["elem"] = lk["elem"] + off
                ls.append(lk)
            off += len(p)
        links = {k: np.concatenate([lk[k] for lk in ls]) for k in ls[0]}
    return WallElements(
        cat("points"),
        cat("normals"),
        cat("weights"),
        cat("factors"),
        cat("jet"),
        inside,
        np.concatenate([p.sampling_normals for p in parts]),
        links,
    )


def _half_space_link_jet(jets: Bc2007Jets, j: int, ax: int, sgn: float, wall: float, dX: float, origin) -> WallElements:
    """Link model of jet ``j`` on the flat wall ``x[ax] = wall`` (fluid on the side ``sgn``), see ``link_jets``."""
    n = np.zeros(jets.n_dim)
    n[ax] = sgn

    def inside(p):
        return (np.asarray(p)[:, ax] - wall) * sgn < 0

    def cut(x0, x1):
        s0, s1 = (x0[:, ax] - wall) * sgn, (x1[:, ax] - wall) * sgn
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((s0 > 0) & (s1 <= 0), s0 / (s0 - s1), np.nan)

    def foot(x):
        f = np.array(x, dtype=float)
        f[:, ax] = wall
        return f, np.broadcast_to(n, f.shape).copy()

    single = Bc2007Jets(
        jets.n_dim,
        jets.object_class,
        jets.centers,
        jets.angles,
        jets.ranges,
        jets.limits,
        jets.normals,
        jets.slot_axes,
        jets.sphere_center,
        jets.sphere_radius,
    )
    lo, hi = single.boxes()
    cells = lattice_cells_in_boxes(single, dX, origin)
    cells = cells[np.all((cells > lo[j] - 2 * dX) & (cells < hi[j] + 2 * dX), axis=1)]
    el = link_jets(single, dX, cells, inside, cut, foot, 9)
    el = el.subset(el.jet == j)
    el.inside = None
    return el


def read_polyline_stl(path: str, n_dim: int = 2) -> np.ndarray:
    """Wall segments (m, 2, 2) of a 2D m-AIA geometry file (first line: number of points, then x y per point)."""
    with open(path) as fh:
        vals = fh.read().split()
    n = int(float(vals[0]))
    xy = np.asarray(vals[1 : 1 + 2 * n], dtype=float).reshape(n, 2)
    return np.stack([xy[:-1], xy[1:]], axis=1)


def read_stl_section(path: str, z: float) -> np.ndarray:
    """Section of a 3D STL surface (ASCII or binary) with the plane ``z``: segments (m, 2, 2) in the x-y plane."""
    with open(path, "rb") as fh:
        raw = fh.read()
    tri = None
    if len(raw) >= 84:
        ntri = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        if 84 + 50 * ntri == len(raw):
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(raw, dtype=rec, count=ntri, offset=84)["v"].astype(float)
    if tri is None:
        txt = raw.decode(errors="replace").split()
        vs = [i for i, t in enumerate(txt) if t == "vertex"]
        tri = np.asarray([[float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3])] for i in vs]).reshape(-1, 3, 3)
    segs = []
    zr = tri[:, :, 2] - z
    for t, s in zip(tri, zr):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (s[a] > 0) != (s[b] > 0):
                lam = s[a] / (s[a] - s[b])
                pts.append(t[a, :2] + lam * (t[b, :2] - t[a, :2]))
        if len(pts) == 2:
            segs.append(pts)
    segs = np.asarray(segs, dtype=float).reshape(-1, 2, 2)
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    return segs[L > 1e-9 * max(L.max(initial=0.0), 1e-300)]


def segments_inside(segs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Point-in-solid test for the closed curves made of ``segs`` (ray casting; unordered segments are fine)."""
    a, b = segs[:, 0], segs[:, 1]

    def inside(p):
        p = np.asarray(p, dtype=float).reshape(-1, p.shape[-1] if np.ndim(p) > 1 else 2)
        x, y = p[:, 0:1], p[:, 1:2]
        cond = (a[None, :, 1] > y) != (b[None, :, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a[None, :, 0] + (y - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (b[None, :, 1] - a[None, :, 1])
        return (np.count_nonzero(cond & (x < xc), axis=1) % 2) == 1

    return inside


def curve_jets(
    jets: Bc2007Jets,
    segs: np.ndarray,
    dX: float,
    per_cell: int = 8,
    cell_offset: float = 0.5,
    z_planes: Optional[np.ndarray] = None,
    span_weight: float = 1.0,
) -> WallElements:
    """Jets on a curved 2D wall (polyline), optionally extruded along z (3D, spanwise-uniform geometry).

    The wall segments near the jets are subdivided (``per_cell`` points per cell length, midpoint rule); the
    normal points into the fluid (outside the closed curve); the factor is that of the cell centre ``cell_offset``
    cells out along the normal. In 3D the elements are repeated on ``z_planes`` (cell-centre planes) with weight
    ``span_weight`` (cells of span per plane).
    """
    lo_b, hi_b = jets.boxes()
    lo = lo_b[:, :2].min(0) - 2 * dX
    hi = hi_b[:, :2].max(0) + 2 * dX
    inside = segments_inside(segs)
    near = np.all((np.minimum(segs[:, 0], segs[:, 1]) < hi) & (np.maximum(segs[:, 0], segs[:, 1]) > lo), axis=1)
    pts, nrm, wts = [], [], []
    for a, b in segs[near]:
        L = np.hypot(*(b - a))
        if L == 0.0:
            continue
        m = max(int(np.ceil(per_cell * L / dX)), 1)
        s = (np.arange(m) + 0.5) / m
        p = a[None, :] + s[:, None] * (b - a)[None, :]
        t = (b - a) / L
        n = np.array([-t[1], t[0]])
        if inside((0.5 * (a + b) + 1e-3 * dX * n)[None, :])[0]:
            n = -n
        pts.append(p)
        nrm.append(np.broadcast_to(n, p.shape))
        wts.append(np.full(m, L / m / dX))
    P, N, W = (np.concatenate(x) for x in (pts, nrm, wts))
    if jets.n_dim == 3:
        zs = np.asarray(z_planes, dtype=float).reshape(-1)
        P = np.concatenate([np.column_stack([P, np.full(len(P), z)]) for z in zs])
        N = np.concatenate([np.column_stack([N, np.zeros(len(N))])] * len(zs))
        W = np.concatenate([W * span_weight] * len(zs))
        inside3 = inside

        def inside(p):  # noqa: F811 - spanwise-uniform solid
            return inside3(np.asarray(p)[:, :2])

    return _assign(jets, P, N, W, np.full(len(P), cell_offset * dX), inside)


def sphere_jets(
    jets: Bc2007Jets,
    dX: float,
    n_theta: int = 8,
    per_cell: float = 1.0,
    cell_offset: float = 0.5,
    max_phi: int = 128,
    keep_all: bool = False,
) -> WallElements:
    """Jets of object class 3 (sphere): quadrature over the band of each jet on the sphere surface.

    Across the band (polar angle about the slot axis): Gauss-Legendre over the exact support of the parabolic profile
    of the cells ``cell_offset`` cells out along the radius; along it (azimuth): midpoint rule over the jet's angular
    range (about ``per_cell`` points per cell, at most ``max_phi``).
    """
    if jets.object_class != 3:
        raise ValueError("sphere_jets: object class 3 only")
    c0, R = jets.sphere_center, jets.sphere_radius
    Rc = R + cell_offset * dX
    xg, wg = np.polynomial.legendre.leggauss(int(n_theta))
    pts, nrm, wts = [], [], []
    for j in range(jets.n_jets):
        ax = int(jets.slot_axes[j])
        hw = min(jets.limits[j, ax], jets.limits[j, ax + 3])
        th_j = np.arccos(np.clip((jets.centers[j, ax] - c0[ax]) / R, -1.0, 1.0))
        # cell at radius Rc and polar angle t has theta_cell = acos(Rc cos t / R): support |R (theta_cell - th_j)| <= hw
        c_lo, c_hi = np.cos(np.clip(th_j + hw / R, 0, np.pi)), np.cos(np.clip(th_j - hw / R, 0, np.pi))
        t_a = np.arccos(np.clip(c_hi * R / Rc, -1, 1))
        t_b = np.arccos(np.clip(c_lo * R / Rc, -1, 1))
        th = 0.5 * (t_a + t_b) + 0.5 * (t_b - t_a) * xg
        wth = 0.5 * (t_b - t_a) * wg
        e = [k for k in range(3) if k != ax]
        vj = jets.centers[j] - c0
        phi_j = np.arctan2(vj[e[1]], vj[e[0]])
        rng = min(jets.ranges[j], 2 * np.pi)
        arc = R * np.sin(th_j) * rng
        nphi = int(min(max(np.ceil(per_cell * arc / dX), 8), max_phi))
        phi = phi_j + (np.arange(nphi) + 0.5) / nphi * rng - 0.5 * rng
        T, F = np.meshgrid(th, phi, indexing="ij")
        WT = np.meshgrid(wth, phi, indexing="ij")[0]
        n = np.zeros(T.shape + (3,))
        n[..., ax] = np.cos(T)
        n[..., e[0]] = np.sin(T) * np.cos(F)
        n[..., e[1]] = np.sin(T) * np.sin(F)
        n = n.reshape(-1, 3)
        pts.append(c0 + R * n)
        nrm.append(n)
        wts.append((R**2 * np.sin(T) * WT * (rng / nphi)).ravel() / dX**2)
    P, N, W = (np.concatenate(x) for x in (pts, nrm, wts))

    def inside(p):
        return np.linalg.norm(np.asarray(p) - c0, axis=1) < R

    if keep_all:
        return WallElements(P, N, W, np.zeros_like(P), np.zeros(len(P), dtype=int), inside)
    return _assign(jets, P, N, W, np.full(len(P), cell_offset * dX), inside)


# ------------------------------------------------------------------------------------------- link-level jet model
def lattice_links(n_dist: int):
    """Non-rest lattice velocities (q-1, nDim) and weights of D2Q9, D3Q19 or D3Q27."""
    if n_dist == 9:
        c = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
        w = np.where(np.abs(c).sum(1) == 1, 1 / 9, 1 / 36)
    elif n_dist in (19, 27):
        c = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
        m = np.abs(c).sum(1)
        if n_dist == 19:
            c, m = c[m <= 2], m[m <= 2]
            w = np.where(m == 1, 1 / 18, 1 / 36)
        else:
            w = np.select([m == 1, m == 2, m == 3], [2 / 27, 1 / 54, 1 / 216])
    else:
        raise ValueError(f"lattice with {n_dist} distributions not supported")
    return c, w


def link_jets(
    jets: Bc2007Jets, dX: float, cells: np.ndarray, inside: Callable, cut: Callable, foot: Callable, n_dist: int
) -> WallElements:
    """Jets on a curved wall, at the level of the lattice links (the moving-wall bounce-back of BC 2007).

    BC 2007 imposes the wall velocity ``u_W`` of a boundary cell (factor of its centre, ``Bc2007Jets.tag``) through
    the moving-wall source ``S = 2 w_k rho c_k . u_W / c_s^2`` of the population it reflects into the fluid along each
    cut link (``interpolatedBounceBackSingleSpecies``, Bouzidi et al. 2001). Which cell writes a link follows the
    solver (``calculateWallDistancesSTD_correctRedundantCuts``): the fluid boundary cell where the link is cut at
    ``q <= 1/2`` (source ``S`` with its own ``u_W``), the solid boundary cell across the link where ``q > 1/2``
    (source ``S / (2 q)`` with the solid cell's ``u_W``). The fluid boundary cells form one wall element each (flux
    vector ``F = 6 sum_k w_k c_k`` of their reflected directions ``c_k``), every solid-cell link one more.

    The interpolation of the bounce-back adds a flow-dependent part to the mass that crosses each cut link per step:
    ``(1 - 2q) [f_k(x + c_k) - f_k(x)]`` (fluid cell writes; populations moving into the wall, from the next cell
    out) and ``(2q - 1) / (2q) [f_k'(x) - f_k(x)]`` (solid cell writes; ``k'`` the reflected direction). It vanishes for
    walls on cell faces (``q = 1/2``). The links are kept in ``links`` so that ``WallJetPower`` can add this part
    from the probed cell values (equilibrium populations).

    Args:
        cells: candidate fluid-side cell centres (n, nDim) near the jets (every fluid boundary cell of the jets must
            be among them; others are ignored).
        inside: point-in-solid test; ``cut(x0, x1)``: fraction in (0, 1] of the first wall crossing on the segment,
            ``nan`` if none; ``foot(x)``: nearest wall point and the wall normal into the fluid.
        n_dist: lattice (9, 19, 27).
    """
    c, wk = lattice_links(n_dist)
    d = jets.n_dim
    opp = np.array([np.flatnonzero(np.all(c == -ck, axis=1))[0] for ck in c])
    cells = np.asarray(cells, dtype=float).reshape(-1, d)
    cells = cells[~inside(cells)]
    nc, nl = len(cells), len(c)
    x0 = np.repeat(cells, nl, axis=0)
    x1 = x0 + np.tile(c, (nc, 1)) * dX
    t = cut(x0, x1).reshape(nc, nl)
    hit = np.isfinite(t) & (t < 1.0)
    bnd = np.any(hit, axis=1)
    cells, t, hit = cells[bnd], t[bnd], hit[bnd]
    nc = len(cells)
    t = np.where(hit & hit[:, opp] & (t <= 0.5), 0.5, t)  # cut in both directions: simple bounce-back (solver)
    tag_f, fac_f = jets.tag(cells)
    # links cut beyond the half link are written by the solid boundary cell across them
    far = hit & (t > 0.5 + 1e-9)
    nb = cells[:, None, :] + c[None, :, :] * dX
    ci, li = np.nonzero(far)
    nb_pts = nb[ci, li]
    solid = inside(nb_pts) if len(nb_pts) else np.zeros(0, dtype=bool)
    near = hit & ~far
    near[ci[~solid], li[~solid]] = True  # (no solid cell across: the fluid cell keeps the link)
    ci, li, nb_pts = ci[solid], li[solid], nb_pts[solid]
    tag_s, fac_s = jets.tag(nb_pts) if len(nb_pts) else (np.zeros(0, int), np.zeros((0, d)))
    src = 6.0 * wk[:, None] * (-c)  # (nl, d): source direction = reflected population, away from the wall
    F_near = np.einsum("nl,ld->nd", near.astype(float), src)
    F_solid = src[li] / (2.0 * t[ci, li])[:, None]
    fp, fn = foot(cells)

    # jet cells: fluid boundary cells with a jet velocity, and fluid cells whose links are written by solid cells
    # with a jet velocity; all cut links of these cells form the jet (also the links whose writer has no jet
    # velocity: their interpolation terms move fluid through the jet's wall as well)
    has_f = (tag_f >= 0) & np.any(fac_f != 0.0, axis=1) & np.any(near, axis=1)
    has_s = (tag_s >= 0) & np.any(fac_s != 0.0, axis=1)
    jet_of = np.full(nc, -1)
    jet_of[ci[has_s]] = tag_s[has_s]
    jet_of[has_f] = tag_f[has_f]
    own = jet_of >= 0
    idx_f = np.flatnonzero(own & np.any(near, axis=1))
    idx_s = np.flatnonzero(own[ci])
    fac_fe = np.where(has_f[idx_f, None], fac_f[idx_f], 0.0)
    fac_se = np.where(has_s[idx_s, None], fac_s[idx_s], 0.0)
    jet_fe = jet_of[idx_f]
    jet_se = np.where(has_s[idx_s], tag_s[idx_s], jet_of[ci[idx_s]])
    F = np.concatenate([F_near[idx_f], F_solid[idx_s]])
    mag = np.linalg.norm(F, axis=1)
    owner = np.concatenate([idx_f, ci[idx_s]])  # fluid cell of every element
    el = WallElements(
        fp[owner],
        F / mag[:, None],
        mag,
        np.concatenate([fac_fe, fac_se]),
        np.concatenate([jet_fe, jet_se]),
        inside,
        fn[owner],
    )
    # links of the elements (for the interpolation terms)
    elem_of_cell = np.full(nc, -1)
    elem_of_cell[idx_f] = np.arange(len(idx_f))
    lc, ll = np.nonzero(near & (elem_of_cell[:, None] >= 0))
    n_lc = len(lc)
    lc = np.concatenate([lc, ci[idx_s]])
    ll = np.concatenate([ll, li[idx_s]])
    el.links = dict(
        elem=np.concatenate([elem_of_cell[lc[:n_lc]], len(idx_f) + np.arange(len(idx_s))]),
        cell=cells[lc],
        up=cells[lc] - c[ll] * dX,  # next cell out along the reflected direction
        c=(-c[ll]).astype(float),  # reflected direction (away from the wall)
        w=wk[ll],
        q=t[lc, ll],
        far=np.arange(len(lc)) >= n_lc,
    )
    return el


def polyline_wall(segs: np.ndarray):
    """``inside``, ``cut`` and ``foot`` of a closed 2D wall made of segments (m, 2, 2)."""
    inside = segments_inside(segs)
    a, b = segs[:, 0], segs[:, 1]
    tng = b - a
    L = np.linalg.norm(tng, axis=1)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], 1) / np.where(L > 0, L, 1.0)[:, None]
    mid = 0.5 * (a + b)
    flip = inside(mid + 1e-6 * np.where(L > 0, L, 1.0).min() * nrm)
    nrm[flip] *= -1.0

    def cut(x0, x1):
        x0, x1 = np.asarray(x0)[:, :2], np.asarray(x1)[:, :2]
        r = x1 - x0
        out = np.full(len(x0), np.nan)
        for s0 in range(0, len(x0), 4096):
            p, rr = x0[s0 : s0 + 4096, None, :], r[s0 : s0 + 4096, None, :]
            den = rr[..., 0] * tng[None, :, 1] - rr[..., 1] * tng[None, :, 0]
            q = a[None] - p
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (q[..., 0] * tng[None, :, 1] - q[..., 1] * tng[None, :, 0]) / den
                uu = (q[..., 0] * rr[..., 1] - q[..., 1] * rr[..., 0]) / den
            ok = (den != 0) & (tt > 0) & (tt <= 1) & (uu >= 0) & (uu <= 1)
            tt = np.where(ok, tt, np.inf).min(axis=1)
            out[s0 : s0 + 4096] = np.where(np.isfinite(tt), tt, np.nan)
        return out

    def foot(x):
        x = np.asarray(x)[:, :2]
        lam = np.clip(np.einsum("nmk,mk->nm", x[:, None, :] - a[None], tng) / np.where(L > 0, L**2, 1.0)[None], 0, 1)
        f = a[None] + lam[..., None] * tng[None]
        k = np.argmin(np.linalg.norm(x[:, None, :] - f, axis=2), axis=1)
        return f[np.arange(len(x)), k], nrm[k]

    return inside, cut, foot


def lattice_cells_in_boxes(jets: Bc2007Jets, dX: float, lattice_origin, margin: float = 2.0, span=None):
    """Finest-level cell centres in the union of the jets' bounding boxes, widened by ``margin`` cells."""
    origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[: jets.n_dim]
    lo_b, hi_b = jets.boxes()
    out = []
    for j in range(jets.n_jets):
        axes = []
        for k in range(min(jets.n_dim, 2)):
            axes.append(_cell_centres(lo_b[j, k] - margin * dX, hi_b[j, k] + margin * dX, origin[k], dX))
        X = np.meshgrid(*axes, indexing="ij")
        out.append(np.stack([g.ravel() for g in X], 1))
    return np.unique(np.round(np.concatenate(out) / dX * 1e6) / 1e6 * dX, axis=0)


def curve_link_jets(
    jets: Bc2007Jets,
    segs: np.ndarray,
    dX: float,
    lattice_origin,
    n_dist: int = 9,
    z_planes: Optional[np.ndarray] = None,
    span_weight: float = 1.0,
) -> WallElements:
    """``link_jets`` on a 2D wall made of segments; in 3D extruded along z (spanwise-uniform geometry).

    In 3D (D3Q27 = D2Q9 x D1Q3) the in-plane sources of the extruded geometry equal those of D2Q9, so the section is
    treated in 2D and repeated on ``z_planes`` (cell-centre planes) with weight ``span_weight`` cells each.
    """
    inside, cut, foot = polyline_wall(segs)
    if jets.n_dim == 2:
        return link_jets(jets, dX, lattice_cells_in_boxes(jets, dX, lattice_origin), inside, cut, foot, n_dist)
    if n_dist != 27:
        raise ValueError("extruded walls in 3D: D3Q27 only (tensor product of D2Q9)")
    j2 = Bc2007Jets(2, jets.object_class, jets.centers[:, :2], jets.angles, jets.ranges, jets.limits[:, [0, 1, 3, 4]])
    el = link_jets(j2, dX, lattice_cells_in_boxes(j2, dX, lattice_origin), inside, cut, foot, 9)
    zs = np.asarray(z_planes, dtype=float).reshape(-1)
    # the z extent of each jet (3D box) decides which planes it covers
    lo_b, hi_b = jets.boxes()
    rows = []
    for z in zs:
        on = (z > lo_b[el.jet, 2]) & (z < hi_b[el.jet, 2])
        rows.append(on)
    on = np.stack(rows)  # (planes, n)
    k = np.nonzero(on)
    P = np.column_stack([el.points[k[1]], zs[k[0]]])

    def ext(a):
        return np.column_stack([a[k[1]], np.zeros(len(k[1]))])

    out = WallElements(
        P,
        ext(el.normals),
        el.weights[k[1]] * span_weight,
        ext(el.factors),
        el.jet[k[1]],
        lambda p: inside(np.asarray(p)[:, :2]),
        ext(el.sampling_normals),
    )
    # links per plane (in-plane D2Q9-equivalent links: the sums over c_z of the D3Q27 weights)
    L = el.links
    new_index = np.full((len(zs), len(el)), -1)
    new_index[k[0], k[1]] = np.arange(len(k[0]))
    planes, lk = np.nonzero(new_index[:, L["elem"]] >= 0)
    z = zs[planes][:, None]
    out.links = dict(
        elem=new_index[planes, L["elem"][lk]],
        cell=np.column_stack([L["cell"][lk], z]),
        up=np.column_stack([L["up"][lk], z]),
        c=np.column_stack([L["c"][lk], np.zeros(len(lk))]),
        w=L["w"][lk] * span_weight,
        q=L["q"][lk],
        far=L["far"][lk],
    )
    return out


def sphere_link_jets(jets: Bc2007Jets, dX: float, lattice_origin, n_dist: int = 27) -> WallElements:
    """``link_jets`` for the jets of object class 3 on the sphere (``lbJetCenterSph``, ``lbJetRadius[0]``)."""
    if jets.object_class != 3:
        raise ValueError("sphere_link_jets: object class 3 only")
    c0, R = jets.sphere_center, jets.sphere_radius
    origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[:3]
    # candidate cells: around a quadrature of the jet bands widened by three cells
    wide = Bc2007Jets(
        3,
        3,
        jets.centers,
        jets.angles,
        jets.ranges + 6 * dX / max(R, 1e-12),
        jets.limits + 3 * dX,
        jets.normals,
        jets.slot_axes,
        c0,
        R,
    )
    band = sphere_jets(
        wide,
        dX,
        n_theta=max(8, int(np.ceil(2 * (wide.limits[:, 0].max()) / dX)) + 2),
        per_cell=1.5,
        cell_offset=0.0,
        max_phi=4096,
        keep_all=True,
    )
    base = np.floor((band.points + dX * band.normals - origin) / dX - 0.5).astype(np.int64)
    off = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T
    idx = np.unique((base[:, None, :] + off[None]).reshape(-1, 3), axis=0)
    cells = origin + (idx + 0.5) * dX
    r = np.linalg.norm(cells - c0, axis=1)
    cells = cells[(r > R) & (r < R + 1.8 * dX)]

    def inside(p):
        return np.linalg.norm(np.asarray(p) - c0, axis=1) < R

    def cut(x0, x1):
        dv = x1 - x0
        a = np.sum(dv * dv, axis=1)
        b = 2 * np.sum(dv * (x0 - c0), axis=1)
        c = np.sum((x0 - c0) ** 2, axis=1) - R**2
        disc = b * b - 4 * a * c
        with np.errstate(invalid="ignore"):
            t = (-b - np.sqrt(disc)) / (2 * a)
        return np.where((disc >= 0) & (c > 0) & (t > 0) & (t <= 1), t, np.nan)

    def foot(x):
        v = np.asarray(x) - c0
        n = v / np.linalg.norm(v, axis=1, keepdims=True)
        return c0 + R * n, n

    return link_jets(jets, dX, cells, inside, cut, foot, n_dist)


# ------------------------------------------------------------------------------------------------------------ model
class WallJetPower:
    """Actuation power of BC-2007 jets given as wall elements (see the module docstring).

    Args:
        elements: ``WallElements`` of all jets (from ``flat_wall_jets``, ``curve_jets`` or
            ``sphere_jets``).
        n_jets: number of jets (actions).
        lattice_origin: origin of the finest-level cell-centre lattice (cell centres at ``origin + (i + 1/2) dX``).
        dX, Ma: finest cell size and Mach number (``U = Ma / sqrt(3)``).
        l_proj: projection length (2D, cells) or area (3D, cells^2) of the environment's drag coefficient.
        wall_distances: distances of the pressure samples from the wall along its normal (cells).
    """

    def __init__(
        self,
        elements: WallElements,
        n_jets: int,
        lattice_origin,
        dX: float,
        Ma: float,
        l_proj: float,
        wall_distances: Sequence[float] = (1.5, 2.5, 3.5, 4.5),
    ):
        if len(elements) == 0:
            raise ValueError("actuation power: no wall cell carries a jet velocity (check the jet geometry)")
        self.el = elements
        self.n_jets = int(n_jets)
        self.n_dim = elements.points.shape[1]
        self.dX = float(dX)
        self.U = float(Ma) / np.sqrt(3.0)
        self.l_proj = float(l_proj)
        self.origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[: self.n_dim]
        self.dist = np.asarray(wall_distances, dtype=float)
        if len(self.dist) < 3 or self.dist.min() < 0.5:
            raise ValueError("wall_distances: at least three distances, the nearest >= 0.5 cells")
        d = self.n_dim
        # elements of the same wall point share their pressure samples
        key = np.round(np.hstack([elements.points / self.dX, elements.sampling_normals]) * 1e6).astype(np.int64)
        _, first, self._site = np.unique(key, axis=0, return_index=True, return_inverse=True)
        self._site = self._site.ravel()
        foot, snrm = elements.points[first], elements.sampling_normals[first]
        smp = foot[:, None, :] + self.dist[None, :, None] * self.dX * snrm[:, None, :]
        f = (smp - self.origin) / self.dX - 0.5
        f_round = np.round(f)
        f = np.where(np.abs(f - f_round) < 1e-4, f_round, f)  # on a cell centre (to grid round-off): one cell
        i0 = np.floor(f).astype(np.int64)
        a = f - i0
        corners = np.array(np.meshgrid(*[[0, 1]] * d, indexing="ij")).reshape(d, -1).T  # (2^d, d)
        idx = i0[:, :, None, :] + corners[None, None, :, :]  # (n, K, 2^d, d)
        w = np.prod(np.where(corners[None, None, :, :] == 1, a[:, :, None, :], 1.0 - a[:, :, None, :]), axis=-1)
        used = w > 1e-12
        cells, inv = np.unique(idx[used], axis=0, return_inverse=True)
        self._cell_index = np.zeros(w.shape, dtype=np.int64)
        self._cell_index[used] = inv.ravel()
        self._w = np.where(used, w, 0.0)
        self._xy = self.origin + (cells + 0.5) * self.dX
        if elements.inside is not None and np.any(elements.inside(self._xy)):
            raise ValueError("actuation power: pressure stencil reaches into the body; increase wall_distances")
        self.probe_coordinates = self._xy.reshape(-1)
        self.n_cells = len(cells)
        self.last_extrapolation_spread = np.nan
        self.last_mass_flux = np.full(self.n_jets, np.nan)  # per jet, incl. the interpolation terms (last sample)
        self.last_flux_correction = np.nan  # interpolation terms / nominal mass flux (all jets)
        # cells of the cut links (bounce-back interpolation terms), probed after the pressure cells
        self._links = elements.links
        if self._links is not None and len(self._links["elem"]):
            lc = np.round(np.concatenate([self._links["cell"], self._links["up"]]) / self.dX * 1e6).astype(np.int64)
            ucells, first, inv = np.unique(lc, axis=0, return_index=True, return_inverse=True)
            inv = inv.ravel()
            nlk = len(self._links["elem"])
            self._link_cell, self._link_up = inv[:nlk], inv[nlk:]
            self._link_xy = np.concatenate([self._links["cell"], self._links["up"]])[first]
            self._link_fluid = ~elements.inside(self._link_xy) if elements.inside is not None else None
            self.probe_coordinates = np.concatenate([self._xy, self._link_xy]).reshape(-1)
        else:
            self._links = None
            self._link_xy = np.zeros((0, d))

    def coefficient(self, p_lattice):
        """Lattice power -> C_P."""
        return 2.0 * np.asarray(p_lattice, dtype=float) / (self.U**3 * self.l_proj)

    def _split(self, probe_data):
        nvar = 4 if self.n_dim == 2 else 5
        dat = np.asarray(probe_data, dtype=float).reshape(-1, nvar)
        if len(dat) != self.n_cells + len(self._link_xy):
            raise ValueError("probe data do not match the model's probe_coordinates")
        rho = dat[: self.n_cells, nvar - 2]
        if not np.isfinite(dat).all():
            raise RuntimeError(
                "actuation power: non-finite probe data (NaN/inf) at the jet-wall probes: the flow solution diverged "
                "(or the restart state is invalid)"
            )
        if np.any(rho <= 0):
            bad = int(np.count_nonzero(rho <= 0))
            if getattr(self, "_valid_state_seen", False):
                # the same probe cells held fluid before: the solver state broke down, the geometry is fine
                raise RuntimeError(
                    f"actuation power: density <= 0 at {bad} of {self.n_cells} jet-wall probe cells (min "
                    f"{rho.min():.3g}) after valid states: the flow solution diverged next to a jet (e.g. sustained "
                    "suction above the stable jet amplitude of the case; see the SquareCylinder notes)"
                )
            raise RuntimeError(
                f"actuation power: probe points outside the fluid domain (zero density at {bad} of {self.n_cells} "
                "jet-wall probe cells); check the jet geometry (lbJetCenter/lbJetRadius/lbJetRanges)"
            )
        self._valid_state_seen = True
        return dat[: self.n_cells], dat[self.n_cells :]

    def wall_state(self, probe_data: np.ndarray, degree: int = 2):
        """Wall pressure and density per element, by least-squares extrapolation along the normal."""
        dat, _ = self._split(probe_data)
        rho, p = dat[:, -2], dat[:, -1]
        ps = np.sum(self._w * p[self._cell_index], axis=-1)  # (n, K)
        rs = np.sum(self._w * rho[self._cell_index], axis=-1)
        V = np.vander(self.dist, degree + 1)
        pinv = np.linalg.pinv(V)[-1]  # value at the wall = constant coefficient
        return (ps @ pinv)[self._site], (rs @ pinv)[self._site]

    def interpolation_flux(self, probe_data) -> np.ndarray:
        """Flow-dependent mass per step through the cut links of each element (bounce-back interpolation, see
        ``link_jets``), from equilibrium populations of the probed cells; zeros without links."""
        out = np.zeros(len(self.el))
        if self._links is None:
            return out
        _, ld = self._split(probe_data)
        d = self.n_dim
        u, rho = ld[:, :d], ld[:, d]
        L = self._links
        c, w, q = L["c"], L["w"], L["q"]

        def feq(k, cdir):  # equilibrium population along cdir at link cells k
            cu = np.sum(cdir * u[k], axis=1)
            return w * rho[k] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * np.sum(u[k] ** 2, axis=1))

        x, up = self._link_cell, self._link_up
        far = q > 0.5
        near_term = (1 - 2 * q) * (feq(up, -c) - feq(x, -c))
        far_term = (2 * q - 1) / (2 * q) * (feq(x, c) - feq(x, -c))
        val = np.where(far, far_term, near_term)
        if self._link_fluid is not None:  # next cell out inside the body (thin parts): no interpolation term
            val = np.where(far | self._link_fluid[up], val, 0.0)
        return np.bincount(L["elem"], weights=val, minlength=len(self.el))

    def _terms(self, p, rho, commands, corr):
        a = np.asarray(commands, dtype=float).reshape(-1)
        if len(a) != self.n_jets:
            raise ValueError(f"expected {self.n_jets} jet commands, got {len(a)}")
        uw = a[self.el.jet][:, None] * self.el.factors
        w = self.el.weights
        vn = np.sum(uw * self.el.normals, axis=1) + corr / (rho * w)  # mass flux / (rho area)
        p_ref = np.sum(w * p) / np.sum(w)
        return vn * (p - p_ref), 0.5 * rho * vn * np.sum(uw**2, axis=1), w, vn

    def point_terms(self, probe_data, commands):
        """Pressure and kinetic power density per element and the element weights (lattice units)."""
        corr = self.interpolation_flux(probe_data)
        p, rho = self.wall_state(probe_data, 2)
        pi_p, pi_k, w, vn = self._terms(p, rho, commands, corr)
        p1, rho1 = self.wall_state(probe_data, 1)
        q_p, q_k, _, _ = self._terms(p1, rho1, commands, corr)
        s2, s1 = np.sum(w * (pi_p + pi_k)), np.sum(w * (q_p + q_k))
        self.last_extrapolation_spread = float(abs(s2 - s1) / max(abs(s2), 1e-300))
        self.last_mass_flux = np.bincount(self.el.jet, weights=rho * vn * w, minlength=self.n_jets)
        nominal = self.mass_flux(commands, rho=1.0)
        self.last_flux_correction = float(np.sum(np.abs(corr)) / max(np.sum(np.abs(nominal)), 1e-300))
        return pi_p, pi_k, w

    def mass_flux(self, commands, rho: float = 1.0) -> np.ndarray:
        """Nominal mass flux of each jet into the fluid (moving-wall sources only; lattice units, per unit span in
        2D) for density ``rho``; ``last_mass_flux`` adds the interpolation terms of the last evaluation."""
        a = np.asarray(commands, dtype=float).reshape(-1)
        vn = np.sum(a[self.el.jet][:, None] * self.el.factors * self.el.normals, axis=1)
        return np.bincount(self.el.jet, weights=rho * vn * self.el.weights, minlength=self.n_jets)

    def evaluate(self, probe_data, commands) -> dict:
        """Per-jet signed C_P and the cost forms (point-wise, pressure and kinetic terms separate)."""
        pi_p, pi_k, w = self.point_terms(probe_data, commands)
        forms = {k: float(self.coefficient(v)) for k, v in transpiration_forms(pi_p, pi_k, w).items()}
        per = self.coefficient(np.bincount(self.el.jet, weights=w * (pi_p + pi_k), minlength=self.n_jets))
        return {"per_actuator": per, "forms": forms}


def geometry_files(geometry_toml: str, bc: int = 2007) -> List[str]:
    """Geometry files of the segments with boundary condition ``bc`` in an m-AIA ``geometry.toml``."""
    import os
    import re

    files, bcs = {}, {}
    with open(geometry_toml) as fh:
        for line in fh:
            line = line.split("#", 1)[0]
            m = re.match(r'\s*filename\.(\d+)\s*=\s*"([^"]+)"', line)
            if m:
                files[int(m.group(1))] = m.group(2)
            m = re.match(r"\s*BC\.(\d+)\s*=\s*(-?\d+)", line)
            if m:
                bcs[int(m.group(1))] = int(m.group(2))
    base = os.path.dirname(os.path.abspath(geometry_toml))
    return [os.path.join(base, files[k]) for k in sorted(files) if bcs.get(k) == bc]
