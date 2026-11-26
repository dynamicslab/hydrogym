"""Actuation power of the m-AIA environments, evaluated from probe data (no solver change needed).

Two actuator models, both in lattice units of the finest level and normalised like the drag of the environment
(``C_P = 2 P / (rho_0 U^3 L_proj)`` with ``U = Ma / sqrt(3)`` and ``L_proj = referenceLength / dX * zLength / dX``,
the projection length of ``MaiaFlowEnv.compute_nondim_coefficients``), so ``C_D`` is the drag power coefficient.

* ``RingTorquePower`` - rotating circular bodies (boundary condition 2008, ``lbControlActions`` = angular
  velocity command). The torque of the fluid on each body comes from the angular-momentum balance of the fluid
  between the body and circles in an annulus at 1.3-1.6 body radii, evaluated from the cell-centre values of the
  finest level (m-AIA samples probes by nearest neighbour, so the model probes the cell centres and interpolates
  itself); the shaft power is ``P = -T Omega`` with ``Omega = command * dX`` per lattice time step (the wall velocity
  of BC 2008 is ``command * (-r_y, r_x)`` with ``r`` in geometry units). The momentum exchange of the rotating wall
  links is not used because it does not resolve the small tangential stress reliably.
* ``SlotJetPower`` - slot jets on circular bodies (boundary condition 2007, object class of the cylinder
  environments): radial wall velocity ``v_n(s) = a (1 - s^2 / h^2)`` across the slot of half width ``h`` (the
  parabolic profile of m-AIA). The power is the flux of pressure work and kinetic energy through the slot
  (Bewley, Moin & Temam 2001): ``P = int v_n (p - p_ref + rho v_n^2 / 2) ds``, with the wall pressure extrapolated
  from the cell-centre values 1.5-4.5 cells out along the wall normal and ``p_ref`` the mean wall pressure over all
  slots of a plane (the zero-net-mass-flux group). ``p_ref`` does not change the signed power; it keeps the absolute
  lattice pressure out of the positive and absolute forms, which are formed point-wise across the slots with the
  pressure and kinetic terms separate (``hydrogym.power.transpiration_forms``).

In 3D both models sample several spanwise planes (rotation axis / jets along z) and scale the per-plane value with
the span. Both sample the flow once, at the end of the action interval (``power_sampling = "end"``).

Accuracy (m-AIA runs of the released cases):

* rotation, pinball Re 100 (D/20): saturated rotation C_P 3.132-3.138 for five annuli between 1.3 and 1.7 radii
  (0.2 %); weak rotation with vortex shedding: time means 0.2874-0.2885 (0.4 %). In unsteady flow a single sample
  also contains the change of the angular momentum of the annulus fluid, so per-step values scatter while their
  means converge (``actuation_power_model_spread`` in the info: spread over the circles);
* jets, cylinder Re 1000 (D/64, slot 13 cells): signed C_P 0.191-0.192 for five sampling stencils (0.8 %), absolute
  form within 4 %; at Re 200 (D/20, slot 4 cells) the exit pressure is under-resolved and the signed power varies by
  about +-10 % with the stencil (``actuation_power_model_spread``: linear vs quadratic wall extrapolation).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from hydrogym.power import power_forms, transpiration_forms

ANNULUS = (1.3, 1.6)


class _ProbeModel:
    """Common geometry and normalisation."""

    def __init__(
        self,
        n_dim: int,
        dX: float,
        Ma: float,
        Re: float,
        reference_length: float,
        z_length: float,
        z_center: float = 0.0,
        n_planes: int = 8,
    ):
        self.n_dim = int(n_dim)
        self.dX = float(dX)
        self.U = float(Ma) / np.sqrt(3.0)
        self.D_cells = float(reference_length) / self.dX
        self.nu = self.U * self.D_cells / float(Re)
        self.span_cells = float(z_length) / self.dX if self.n_dim == 3 else 1.0
        self.l_proj = self.D_cells * self.span_cells
        if self.n_dim == 3:
            n = max(int(n_planes), 1)
            self.z_planes = float(z_center) + (np.arange(n) + 0.5) / n * float(z_length) - 0.5 * float(z_length)
        else:
            self.z_planes = np.zeros(1)

    def _with_planes(self, xy: np.ndarray) -> np.ndarray:
        """(n, 2) in-plane points -> flat probe coordinate list over all planes (geometry units)."""
        if self.n_dim == 2:
            return xy.reshape(-1)
        pts = [np.column_stack([xy, np.full(len(xy), z)]) for z in self.z_planes]
        return np.concatenate(pts).reshape(-1)

    def _split(self, data: np.ndarray, n_points: int) -> np.ndarray:
        """Probe data -> (planes, points, vars); checks that every probe lies in the fluid domain."""
        nvar = 4 if self.n_dim == 2 else 5
        d = np.asarray(data, dtype=float).reshape(len(self.z_planes), n_points, nvar)
        if not np.isfinite(d).all():
            raise RuntimeError(
                "actuation power: non-finite probe data (NaN/inf): the solver state is invalid (e.g. a truncated "
                "restart file or a diverged run)"
            )
        rho = d[..., 2] if self.n_dim == 2 else d[..., 3]
        if np.any(rho <= 0):
            raise RuntimeError(
                "actuation power: probe points outside the fluid domain (zero density returned); check the body "
                "geometry (lbJetCenter/lbJetRadius) and, in 3D, the spanwise planes"
            )
        return d

    def coefficient(self, p_lattice: np.ndarray) -> np.ndarray:
        """Lattice power (per unit span in 2D, total in 3D) -> C_P."""
        return 2.0 * np.asarray(p_lattice, dtype=float) / (self.U**3 * self.l_proj)


class RingTorquePower(_ProbeModel):
    """Shaft power of rotating circular bodies from the flow around them (see module docstring).

    m-AIA returns probe values by nearest-neighbour sampling (its default ``interpolationOrder``), so a probe gives the
    value of the cell centre nearest to it. The model therefore probes exactly the cell centres of the finest level
    in an annulus around each body, takes the velocity gradients on that lattice (second-order central differences),
    interpolates bilinearly to circles and evaluates on each circle

        T(r) = r^2 oint [tau_rt - rho u_r u_t] dtheta,
        tau_rt = rho nu [(d_y u + d_x v) cos 2theta + (d_y v - d_x u) sin 2theta],

    the torque of the fluid outside the circle on the fluid inside (the pressure drops out on a circle). For a
    statistically steady flow T does not depend on r; the torque is the mean over ``n_radii`` circles in the annulus
    and its spread over the circles is kept in ``last_spread`` (relative) as a check. The annulus must lie on the
    finest grid level and must not contain another body.

    Args:
        centers: Centres of the rotating bodies, geometry units, shape (n_bodies, 2).
        radii: Their radii, geometry units.
        lattice_origin: (x0, y0) with the finest-level cell centres at ``x0 + (i + 1/2) dX`` (from the grid file:
            ``centerOfGravity - lengthLevel0 / 2``).
        axis_signs: +1 / -1 per body for rotation about +z / -z (3D); 2D: +1 (counter-clockwise positive).
        annulus: Inner and outer circle radius in body radii.
        n_radii, n_theta: Circles in the annulus and points per circle.
        other_bodies: Further (centre x, centre y, radius) whose cells are never probed (e.g. non-rotating bodies).
        other arguments: see ``_ProbeModel`` (from the environment's properties).
    """

    def __init__(
        self,
        centers: Sequence[Sequence[float]],
        radii: Sequence[float],
        lattice_origin: Sequence[float],
        axis_signs: Optional[Sequence[float]] = None,
        annulus: Sequence[float] = ANNULUS,
        n_radii: int = 9,
        n_theta: int = 288,
        other_bodies: Sequence[Sequence[float]] = (),
        **kw,
    ):
        super().__init__(**kw)
        self.centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        nb = len(self.centers)
        self.radii = np.broadcast_to(np.asarray(radii, dtype=float).reshape(-1), (nb,)).copy()
        self.axis_signs = np.ones(nb) if axis_signs is None else np.asarray(axis_signs, dtype=float).reshape(nb)
        self.origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[:2]
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.n_radii = int(n_radii)
        if not 1.0 < self.annulus[0] < self.annulus[1]:
            raise ValueError("annulus: 1 < inner < outer (in body radii)")
        if np.any(self.annulus[0] * self.radii - 2.5 * self.dX < self.radii + 0.5 * self.dX - 1e-9 * self.dX):
            raise ValueError(
                "annulus: the inner circle needs 3 fluid cells to the wall (gradient and interpolation stencil)"
            )
        bodies = [(c[0], c[1], R) for c, R in zip(self.centers, self.radii)]
        bodies += [tuple(float(x) for x in b) for b in other_bodies]
        self.theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
        self.patches = []  # per body: (i0, j0, ni, nj, flat indices of the probed cells in the patch)
        pts = []
        h = self.dX
        for b, (c, R) in enumerate(zip(self.centers, self.radii)):
            r_in, r_out = self.annulus[0] * R - 2.5 * h, self.annulus[1] * R + 2.5 * h
            i0 = int(np.floor((c[0] - r_out - self.origin[0]) / h - 0.5))
            j0 = int(np.floor((c[1] - r_out - self.origin[1]) / h - 0.5))
            ni = nj = int(np.ceil(2 * r_out / h)) + 3
            xc = self.origin[0] + (i0 + np.arange(ni) + 0.5) * h
            yc = self.origin[1] + (j0 + np.arange(nj) + 0.5) * h
            X, Y = np.meshgrid(xc, yc, indexing="ij")
            r = np.hypot(X - c[0], Y - c[1])
            use = (r >= r_in) & (r <= r_out)
            for k, (bx, by, bR) in enumerate(bodies):
                if k != b:
                    too_close = np.hypot(X - bx, Y - by) <= bR + 0.5 * h
                    if np.any(too_close & (r <= self.annulus[1] * R + 2.5 * h + 1e-9 * h)):
                        raise ValueError(f"annulus of body {b} reaches body {k}; reduce the outer radius")
                    use &= ~too_close
            idx = np.flatnonzero(use.ravel())
            self.patches.append((i0, j0, ni, nj, idx))
            pts.append(np.column_stack([X.ravel()[idx], Y.ravel()[idx]]))
        self._counts = [len(p) for p in pts]
        self._xy = np.concatenate(pts)
        self.probe_coordinates = self._with_planes(self._xy)
        self.last_spread = np.full(nb, np.nan)

    def _torque_one(self, b: int, u: np.ndarray, v: np.ndarray, rho: np.ndarray):
        """Mean torque (lattice units, per unit span) and relative spread over the circles, for one body/plane."""
        i0, j0, ni, nj, idx = self.patches[b]
        h = self.dX
        U = np.full(ni * nj, np.nan)
        V = np.full(ni * nj, np.nan)
        Rh = np.full(ni * nj, np.nan)
        U[idx], V[idx], Rh[idx] = u, v, rho
        U, V, Rh = U.reshape(ni, nj), V.reshape(ni, nj), Rh.reshape(ni, nj)
        # gradients per cell (lattice units: cell size 1)
        dudx, dudy = np.gradient(U, axis=0), np.gradient(U, axis=1)
        dvdx, dvdy = np.gradient(V, axis=0), np.gradient(V, axis=1)
        c, R = self.centers[b], self.radii[b]
        radii = np.linspace(self.annulus[0] * R, self.annulus[1] * R, self.n_radii)
        torques = []
        ct, st = np.cos(self.theta), np.sin(self.theta)
        c2, s2 = np.cos(2 * self.theta), np.sin(2 * self.theta)
        for r in radii:
            fx = (c[0] + r * ct - self.origin[0]) / h - 0.5 - i0
            fy = (c[1] + r * st - self.origin[1]) / h - 0.5 - j0
            ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
            ax, ay = fx - ix, fy - iy

            def bil(F):
                return (
                    (1 - ax) * (1 - ay) * F[ix, iy]
                    + ax * (1 - ay) * F[ix + 1, iy]
                    + (1 - ax) * ay * F[ix, iy + 1]
                    + ax * ay * F[ix + 1, iy + 1]
                )

            uu, vv, rr = bil(U), bil(V), bil(Rh)
            gxu, gyu, gxv, gyv = bil(dudx), bil(dudy), bil(dvdx), bil(dvdy)
            if not np.all(np.isfinite([uu, vv, rr, gxu, gyu, gxv, gyv])):
                raise RuntimeError("actuation power: circle outside the probed cells (check annulus and geometry)")
            ur = uu * ct + vv * st
            ut = -uu * st + vv * ct
            tau = rr * self.nu * ((gyu + gxv) * c2 + (gyv - gxu) * s2)
            rc = r / h
            torques.append(rc**2 * np.sum(tau - rr * ur * ut) * (2 * np.pi / len(self.theta)))
        torques = np.asarray(torques)
        mean = float(torques.mean())
        spread = float(np.ptp(torques) / abs(mean)) if mean != 0 else np.inf
        return mean, spread

    def per_body(self, probe_data: np.ndarray, commands: Sequence[float]) -> np.ndarray:
        """C_P of each body from the probe data at ``probe_coordinates`` and the angular-velocity commands."""
        nb = len(self.centers)
        d = self._split(probe_data, int(sum(self._counts)))
        rho_col = 2 if self.n_dim == 2 else 3
        torque = np.zeros(nb)
        spread = np.zeros(nb)
        for k in range(d.shape[0]):  # spanwise planes
            off = 0
            for b in range(nb):
                n = self._counts[b]
                blk = d[k, off : off + n]
                off += n
                t, sp = self._torque_one(b, blk[:, 0], blk[:, 1], blk[:, rho_col])
                torque[b] += t / d.shape[0]
                spread[b] = max(spread[b], sp)
        self.last_spread = spread
        torque *= self.span_cells  # per unit span -> total (2D: unit span)
        omega = np.asarray(commands, dtype=float).reshape(-1) * self.dX * self.axis_signs
        return self.coefficient(-torque * omega)

    def evaluate(self, probe_data: np.ndarray, commands: Sequence[float]) -> dict:
        """Per-body C_P and the cost forms (one motor per rigid body: forms over the bodies)."""
        per = self.per_body(probe_data, commands)
        return {"per_actuator": per, "forms": power_forms(per)}


class SlotJetPower(_ProbeModel):
    """Power of slot jets on circular bodies (see module docstring).

    The exit pressure is taken at the wall: the model probes the finest-level cell centres around each slot (m-AIA
    samples probes by nearest neighbour), interpolates the pressure bilinearly along the wall normal at
    ``wall_distances`` (default 1.5-4.5 cells) from each Gauss point of the slot and extrapolates it to the wall with a
    quadratic least-squares fit. The relative change of the signed power between the quadratic and a linear
    extrapolation is kept in ``last_extrapolation_spread`` as an error estimate.

    Args:
        body_centers: Centre of the body each jet sits on, geometry units, shape (n_jets, 2).
        body_radii: Radius of that body, geometry units, shape (n_jets,).
        angles: Jet angle (radians from +x), shape (n_jets,).
        widths: Slot width ``lbJetRanges`` (geometry units), shape (n_jets,).
        lattice_origin: (x0, y0) of the finest-level cell-centre lattice (see ``lattice_origin_from_grid``).
        n_slot: Gauss-Legendre points across each slot.
        wall_distances: Distances of the pressure samples from the wall along its normal, in finest cells.
        other arguments: see ``_ProbeModel``.
    """

    def __init__(
        self,
        body_centers,
        body_radii,
        angles,
        widths,
        lattice_origin,
        n_slot: int = 12,
        wall_distances: Sequence[float] = (1.5, 2.5, 3.5, 4.5),
        **kw,
    ):
        super().__init__(**kw)
        self.body_centers = np.asarray(body_centers, dtype=float).reshape(-1, 2)
        nj = len(self.body_centers)
        self.body_radii = np.broadcast_to(np.asarray(body_radii, dtype=float).reshape(-1), (nj,)).copy()
        self.angles = np.asarray(angles, dtype=float).reshape(-1)
        self.half_widths = 0.5 * np.broadcast_to(np.asarray(widths, dtype=float).reshape(-1), (nj,))
        self.origin = np.asarray(lattice_origin, dtype=float).reshape(-1)[:2]
        self.dist = np.asarray(wall_distances, dtype=float)
        if len(self.dist) < 3 or self.dist.min() < 1.5:
            raise ValueError("wall_distances: at least three distances, the nearest >= 1.5 cells")
        xg, wg = np.polynomial.legendre.leggauss(int(n_slot))
        self.s_unit, self.w_unit = xg, wg  # on [-1, 1]
        h = self.dX
        self.samples = []  # per jet: sample points (n_slot, n_dist, 2)
        self.patches = []  # per jet: (i0, j0, ni, nj, flat indices of probed cells)
        pts = []
        for c, R, th, hw in zip(self.body_centers, self.body_radii, self.angles, self.half_widths):
            phi = th + np.arcsin(np.clip(xg * hw / R, -1.0, 1.0))
            nrm = np.stack([np.cos(phi), np.sin(phi)], 1)
            foot = c[None, :] + R * nrm
            smp = foot[:, None, :] + self.dist[None, :, None] * h * nrm[:, None, :]
            self.samples.append(smp)
            lo, hi = smp.reshape(-1, 2).min(0) - 2 * h, smp.reshape(-1, 2).max(0) + 2 * h
            i0 = int(np.floor((lo[0] - self.origin[0]) / h - 0.5))
            j0 = int(np.floor((lo[1] - self.origin[1]) / h - 0.5))
            ni = int(np.ceil((hi[0] - lo[0]) / h)) + 3
            nj_ = int(np.ceil((hi[1] - lo[1]) / h)) + 3
            X, Y = np.meshgrid(
                self.origin[0] + (i0 + np.arange(ni) + 0.5) * h,
                self.origin[1] + (j0 + np.arange(nj_) + 0.5) * h,
                indexing="ij",
            )
            use = np.hypot(X - c[0], Y - c[1]) >= R + 0.5 * h
            idx = np.flatnonzero(use.ravel())
            self.patches.append((i0, j0, ni, nj_, idx))
            pts.append(np.column_stack([X.ravel()[idx], Y.ravel()[idx]]))
        self._counts = [len(p) for p in pts]
        self._xy = np.concatenate(pts)
        self.probe_coordinates = self._with_planes(self._xy)
        self.last_extrapolation_spread = np.nan

    def _bilinear(self, j: int, F: np.ndarray, pts: np.ndarray) -> np.ndarray:
        i0, j0 = self.patches[j][0], self.patches[j][1]
        fx = (pts[..., 0] - self.origin[0]) / self.dX - 0.5 - i0
        fy = (pts[..., 1] - self.origin[1]) / self.dX - 0.5 - j0
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - ix, fy - iy
        val = (
            (1 - ax) * (1 - ay) * F[ix, iy]
            + ax * (1 - ay) * F[ix + 1, iy]
            + (1 - ax) * ay * F[ix, iy + 1]
            + ax * ay * F[ix + 1, iy + 1]
        )
        if not np.all(np.isfinite(val)):
            raise RuntimeError("actuation power: jet pressure sample outside the probed cells")
        return val

    def wall_state(self, probe_data: np.ndarray, degree: int = 2):
        """Wall pressure and density at the Gauss points, shapes (planes, jets, points), by extrapolation."""
        nj, ns = len(self.body_centers), len(self.s_unit)
        d = self._split(probe_data, int(sum(self._counts)))
        rho_col, p_col = (2, 3) if self.n_dim == 2 else (3, 4)
        p_w = np.zeros((d.shape[0], nj, ns))
        r_w = np.zeros_like(p_w)
        V = np.vander(self.dist, degree + 1)  # least-squares polynomial in the wall distance, value at 0 = last coef
        for k in range(d.shape[0]):
            off = 0
            for j in range(nj):
                i0, j0, ni, nj_, idx = self.patches[j]
                n = self._counts[j]
                blk = d[k, off : off + n]
                off += n
                P = np.full(ni * nj_, np.nan)
                Rh = np.full(ni * nj_, np.nan)
                P[idx], Rh[idx] = blk[:, p_col], blk[:, rho_col]
                ps = self._bilinear(j, P.reshape(ni, nj_), self.samples[j])  # (ns, ndist)
                rs = self._bilinear(j, Rh.reshape(ni, nj_), self.samples[j])
                p_w[k, j] = np.linalg.lstsq(V, ps.T, rcond=None)[0][-1]
                r_w[k, j] = np.linalg.lstsq(V, rs.T, rcond=None)[0][-1]
        return p_w, r_w

    def _terms(self, p, rho, commands):
        nj = len(self.body_centers)
        a = np.asarray(commands, dtype=float).reshape(1, nj, 1)
        vn = a * (1.0 - self.s_unit**2)[None, None, :]  # parabolic profile at the Gauss points
        # Reference pressure: the area mean of the wall pressure over all slots of a plane (the zero-net-mass-flux
        # group), the p - p_bar of Bewley et al. (2001) and of the NekRS backend. It does not change the signed total
        # of a zero-net-mass-flux set of jets; it keeps the absolute lattice pressure out of the positive and absolute
        # forms.
        wa = np.broadcast_to(self.w_unit[None, None, :] * (self.half_widths / self.dX)[None, :, None], p.shape)
        p_ref = np.sum(p * wa, axis=(1, 2), keepdims=True) / np.sum(wa, axis=(1, 2), keepdims=True)
        pi_p = vn * (p - p_ref)
        pi_k = 0.5 * rho * vn**3
        w = wa * (self.span_cells / len(self.z_planes))
        return pi_p, pi_k, w

    def point_terms(self, probe_data: np.ndarray, commands: Sequence[float]):
        """Point-wise pressure and kinetic power densities and surface weights, shapes (planes, jets, points).

        ``sum(w * (pi_p + pi_k))`` is the lattice power of all jets (per unit span in 2D).
        """
        p, rho = self.wall_state(probe_data, degree=2)
        pi_p, pi_k, w = self._terms(p, rho, commands)
        p1, rho1 = self.wall_state(probe_data, degree=1)
        q_p, q_k, _ = self._terms(p1, rho1, commands)
        s2, s1 = np.sum(w * (pi_p + pi_k)), np.sum(w * (q_p + q_k))
        self.last_extrapolation_spread = float(abs(s2 - s1) / max(abs(s2), 1e-300))
        return pi_p, pi_k, w

    def per_jet(self, probe_data: np.ndarray, commands: Sequence[float]) -> np.ndarray:
        """Signed C_P of each jet from the probe data and the jet velocity commands (lattice units)."""
        pi_p, pi_k, w = self.point_terms(probe_data, commands)
        return self.coefficient(np.sum(w * (pi_p + pi_k), axis=(0, 2)))

    def evaluate(self, probe_data: np.ndarray, commands: Sequence[float]) -> dict:
        """Per-jet signed C_P and the cost forms, formed point-wise with separate pressure and kinetic terms."""
        pi_p, pi_k, w = self.point_terms(probe_data, commands)
        lattice = transpiration_forms(pi_p.ravel(), pi_k.ravel(), w.ravel())
        forms = {k: float(self.coefficient(v)) for k, v in lattice.items()}
        per = self.coefficient(np.sum(w * (pi_p + pi_k), axis=(0, 2)))
        return {"per_actuator": per, "forms": forms}


def read_cdf_global_attributes(path: str) -> dict:
    """Global attributes of a classic-format NetCDF file (CDF-1, CDF-2 or CDF-5, as written by m-AIA's PnetCDF).

    Only the header is read; no NetCDF library is needed.
    """
    import struct

    types = {
        1: ("b", 1),
        2: ("c", 1),
        3: ("h", 2),
        4: ("i", 4),
        5: ("f", 4),
        6: ("d", 8),
        7: ("B", 1),
        8: ("H", 2),
        9: ("I", 4),
        10: ("q", 8),
        11: ("Q", 8),
    }
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    if head[:3] != b"CDF" or head[3] not in (1, 2, 5):
        raise ValueError(f"{path}: not a classic NetCDF file")
    big = head[3] == 5  # CDF-5: 8-byte counts
    pos = 4

    def count():
        nonlocal pos
        fmt, n = (">q", 8) if big else (">i", 4)
        val = struct.unpack_from(fmt, head, pos)[0]
        pos += n
        return val

    def tag():
        nonlocal pos
        val = struct.unpack_from(">i", head, pos)[0]
        pos += 4
        return val

    def name():
        nonlocal pos
        n = count()
        val = head[pos : pos + n].decode()
        pos += (n + 3) // 4 * 4
        return val

    count()  # numrecs
    if tag() == 0x0A:  # dimensions
        for _ in range(count()):
            name()
            count()
    else:
        count()
    attrs = {}
    if tag() == 0x0C:  # global attributes
        for _ in range(count()):
            key = name()
            nc_type = tag()
            n = count()
            code, size = types[nc_type]
            raw = head[pos : pos + n * size]
            pos += (n * size + 3) // 4 * 4
            if code == "c":
                attrs[key] = raw.decode(errors="replace").rstrip("\x00")
            else:
                vals = struct.unpack(">" + code * n, raw)
                attrs[key] = vals[0] if n == 1 else list(vals)
    return attrs


def lattice_origin_from_grid(grid_file: str, dX: float, n_dim: int) -> np.ndarray:
    """Origin of the finest-level cell-centre lattice (cell centres at ``origin + (i + 1/2) dX``).

    The root cell of the m-AIA grid is the cube of side ``lengthLevel0`` centred at ``centerOfGravity``.
    """
    att = read_cdf_global_attributes(grid_file)
    cog = np.asarray(att["centerOfGravity"], dtype=float).reshape(-1)[:n_dim]
    l0 = float(att["lengthLevel0"])
    levels = np.log2(l0 / dX)
    if abs(levels - round(levels)) > 1e-6:
        raise ValueError(f"{grid_file}: lengthLevel0 {l0} is not dX {dX} times a power of two")
    return cog - 0.5 * l0


def grid_finest_cell_size(grid_file: str) -> float:
    """Finest cell size of an m-AIA grid file: ``lengthLevel0 / 2**maxLevel`` (from the header only)."""
    att = read_cdf_global_attributes(grid_file)
    return float(att["lengthLevel0"]) / 2 ** int(att["maxLevel"])


def check_cell_size(grid_file: str, dX: float, source: str = "the property file", rtol: float = 1e-6) -> float:
    """Raise ValueError if the grid's finest cell size differs from ``dX``; return the grid's cell size.

    HydroGym computes ``dX = reductionFactor * domainLength / 2**maxRfnmntLvl`` from the property file and uses it
    to normalise forces, power and probe positions. A property file whose ``domainLength`` does not match the grid
    would otherwise give silently wrong coefficients.
    """
    dx_grid = grid_finest_cell_size(grid_file)
    if abs(dx_grid / float(dX) - 1.0) > rtol:
        raise ValueError(
            f"{grid_file}: finest cell size {dx_grid:.6g} (lengthLevel0 / 2^maxLevel) differs from "
            f"dX = {float(dX):.6g} "
            f"computed from {source} (reductionFactor * domainLength / 2^maxRfnmntLvl); fix domainLength, "
            "reductionFactor or maxRfnmntLvl in properties_run.toml so that they match the grid"
        )
    return dx_grid


def jets_on_bodies(props: dict, n_jets: int) -> Optional[tuple]:
    """Jet geometry of a jet-actuated circular-body environment from its properties, or None if not applicable.

    m-AIA lists ``lbJetCenter`` per jet as the centre of the body the jet sits on (it adds ``lbJetRadius`` along
    ``lbJetAngles`` itself), with ``lbJetRanges`` the slot width.
    """
    try:
        n_dim = int(props["nDim"])
        centers = np.asarray(props["lbJetCenter"], dtype=float).reshape(n_jets, n_dim)[:, :2]
        radii = np.broadcast_to(np.asarray(props["lbJetRadius"], dtype=float).reshape(-1), (n_jets,))
        angles = np.asarray(props["lbJetAngles"], dtype=float).reshape(n_jets)
        widths = np.broadcast_to(np.asarray(props["lbJetRanges"], dtype=float).reshape(-1), (n_jets,))
    except (KeyError, ValueError):
        return None
    return centers, radii, angles, widths


def rotating_bodies(props: dict, n_bodies: int) -> Optional[tuple]:
    """(in-plane centres, radii, centre z, axis sign) of the rotating bodies, or None if not applicable.

    In 2D, BC 2008 rotates counter-clockwise for a positive command; in 3D about ``lbRotationAxis`` (z only here).
    """
    try:
        n_dim = int(props["nDim"])
        centers = np.asarray(props["lbJetCenter"], dtype=float).reshape(n_bodies, n_dim)
        radii = np.broadcast_to(np.asarray(props["lbJetRadius"], dtype=float).reshape(-1), (n_bodies,))
    except (KeyError, ValueError):
        return None
    signs = np.ones(n_bodies)
    if n_dim == 3:
        axis = np.asarray(props.get("lbRotationAxis", [0.0, 0.0, 1.0] * n_bodies), dtype=float).reshape(n_bodies, 3)
        if not np.allclose(np.abs(axis[:, 2]), 1.0) or not np.allclose(axis[:, :2], 0.0):
            return None  # the ring balance is implemented for rotation about z only
        signs = np.sign(axis[:, 2])
    return centers[:, :2], radii, (centers[:, 2] if n_dim == 3 else np.zeros(n_bodies)), signs


ROTATION_STEADY_TOL = 0.1


def rotation_change(previous: Optional[Sequence[float]], commands: Sequence[float], bound: float) -> float:
    """Largest change of the rotation command over one action, relative to the actuation bound.

    The ring torque of ``RingTorquePower`` assumes that the angular momentum of the fluid in the annulus is
    steady at the sampling instant. That holds for steady or slowly varying rotation; when the command changes by a
    sizeable fraction of the bound within one action (e.g. bang-bang reversal), the end-of-action sample also contains
    the change of the annulus angular momentum and can even become negative. Values above ``ROTATION_STEADY_TOL``
    mark such samples as not valid (``info["actuation_power_valid"] = False``).
    """
    c = np.asarray(commands, dtype=float).reshape(-1)
    p = np.zeros_like(c) if previous is None else np.asarray(previous, dtype=float).reshape(-1)
    return float(np.max(np.abs(c - p)) / abs(bound)) if bound else float("inf")


def zero_power(n: int) -> List[float]:
    """Per-actuator zeros (environment without actuation at reset)."""
    return [0.0] * int(n)
