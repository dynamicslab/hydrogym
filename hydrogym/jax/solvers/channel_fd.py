"""Finite-difference solver for the turbulent channel flow (staggered wall-normal ghost cell, fractional step).

The incompressible Navier-Stokes equations in a plane channel (periodic in the streamwise ``x`` and spanwise ``y``
direction, no-slip walls at ``z = 0`` and ``z = Lz``) are advanced with second-order central differences on a
cell-centred grid, an explicit Runge-Kutta scheme and a projection (pressure Poisson) step after every Runge-Kutta
stage. The discretisation follows the three-dimensional channel code of Vuorinen and Keskinen,
"DNSLab: A gateway to turbulent flow simulation in Matlab", Comput. Phys. Commun. 203 (2016) 278-289 (module
``Channel3dLab``), re-implemented here with matrix-free stencils in JAX; the operators reproduce the sparse matrices
of that code to round-off.

Axis convention
---------------
Arrays have shape ``(nx, ny, nz)``: axis 0 is the streamwise direction ``x`` (length ``lx``), axis 1 the spanwise
direction ``y`` (length ``ly``), axis 2 the wall-normal direction ``z`` (height ``lz``). The velocity components are
``u`` (streamwise), ``v`` (spanwise) and ``w`` (wall-normal). DNSLab stores its fields as ``(N1, N2, N3)`` with
``N1`` = span, ``N2`` = length and ``N3`` = wall-normal (plus one ghost cell on each side), so a DNSLab array
``F[i1, i2, i3]`` is the array ``f[i2, i1, i3]`` here, i.e. the first two axes are swapped; its velocity components
``U``, ``V``, ``W`` are the streamwise, spanwise and wall-normal components, identical to ``u``, ``v``, ``w`` here.
DNSLab vectors are column-major (Fortran order) flattenings of ``(N1, N2, N3-2)``.

Discretisation
--------------
* ``x`` and ``y`` are uniform and periodic; ``z`` is stretched with a ``tanh`` map (``ctanh``). The ``nz`` interior
  cells are bounded by the faces ``zf`` (``zf[0] = 0``, ``zf[-1] = lz``); a ghost cell of the width of the adjacent
  wall cell is added below and above.
* Velocity derivatives ``dx, dy, dz, dxx, dyy, dzz``: no-slip walls via the ghost value ``-f_0`` (zero wall value
  halfway between the ghost and the first cell). Pressure derivatives ``dxp, dyp, dzp`` and the Poisson operator
  ``poisson``: zero normal gradient at the walls (ghost value ``+p_0``). The pressure is pinned to zero on a single
  bottom-wall face (ghost value ``-p_0`` in that one cell).
* Convection in skew-symmetric form, ``0.5 * (d(u_j u_i)/dx_j + u_j du_i/dx_j)``; the flux derivative in the
  wall-normal direction uses the pressure-type (Neumann) stencil ``dzp``.
* Time integration: explicit Runge-Kutta (``rk_order`` 1 to 4, default the classical fourth-order scheme). Every
  stage velocity (and the final one) is projected: ``div = dx u + dy v + dz w``, ``poisson(p) = div``,
  ``u <- u - dxp p`` (and likewise for ``v, w``). The Poisson solve is warm-started with the previous pressure.
* The time step is set dynamically from a target Courant number ``cfl`` with the Courant sum of DNSLab,
  ``max(u/dx + v/dy + w/dz_cell)`` (signed, no absolute values; ``courant="abs"`` switches to absolute values).
* The flow is driven by the body force ``gx = 8 nu^2 Re_tau^2 / H^3`` in the streamwise direction.

Pressure solve
--------------
``poisson_solver="direct"`` (default) solves the discrete problem exactly, with a Fourier transform in ``x`` and ``y`` and a
tridiagonal solve in ``z`` (the Poisson operator is separable). It is the solver of the default 32-bit precision and
runs about 10 to 20 times faster than BiCGStab.
``poisson_solver="bicgstab"`` (float64 only) uses ``lineax.BiCGStab`` with a matrix-free ``FunctionLinearOperator``
(differentiable by implicit differentiation) and the stopping criterion ``||r||_2 <= pressure_tol * ||b||_2``;
``jax.scipy.sparse.linalg.bicgstab`` (same criterion, differentiable) is used instead if ``lineax`` is not installed (the
number of iterations is then not reported, ``-1``). The BiCGStab iteration is not reliable for this (strongly anisotropic,
negative definite) operator in 32-bit precision: lineax's breakdown test stops it after a few steps, and the ``jax.scipy``
iteration stagnates or breaks down (an infinite step, then NaN) on a fraction of the solves, which in a controlled flow
led to a divergent or non-finite velocity; the combination is therefore rejected. Float64 requires
``jax.config.update("jax_enable_x64", True)`` before use; this module does not set it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

try:  # lineax is optional: a jax.scipy fallback is used when it is missing
    import lineax as _lx
except Exception:  # pragma: no cover - exercised only without lineax
    _lx = None

_EPS_TINY = 1e-300

if _lx is not None:

    class _BiCGStab(_lx.BiCGStab):
        """``lineax.BiCGStab`` that returns the exact solution (zero) for a zero right-hand side.

        The stock solver divides ``0 / 0`` in its first update and returns NaN. A zero right-hand side occurs in the
        adjoint (transposed) solve whenever the cotangent of the pressure vanishes, and in the forward solve of a
        field that is already free of divergence.
        """

        def compute(self, state, vector, options):
            sol, result, stats = super().compute(state, vector, options)
            return jnp.where(jnp.any(vector != 0), sol, jnp.zeros_like(sol)), result, stats


def _two_norm(x):
    """Euclidean norm of an array (the norm of the BiCGStab stopping test)."""
    return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))

#######################################################################################
#                                                                                     #
#                                  CONFIGURATION                                      #
#                                                                                     #
#######################################################################################


@dataclasses.dataclass(frozen=True)
class ChannelFDConfig:
    """Parameters of the finite-difference channel.

    Attributes:
        nx, ny, nz: Cells in the streamwise, spanwise and wall-normal direction.
        lx, ly, lz: Domain size (streamwise, spanwise, height ``H``).
        nu: Kinematic viscosity.
        re_tau: Friction Reynolds number that sets the body force ``8 nu^2 Re_tau^2 / H^3`` (``H`` = ``lz``).
        body_force: Streamwise body force; ``None`` takes the value from ``re_tau``.
        ctanh: Wall-normal stretching factor (smaller = more uniform).
        cfl: Target maximum Courant number of the dynamic time step.
        courant: ``"dnslab"`` (signed sum of the three directional Courant numbers) or ``"abs"`` (absolute values).
        rk_order: Order of the explicit Runge-Kutta scheme (1 to 4).
        pressure_tol: Relative residual tolerance of the BiCGStab pressure solve.
        pressure_maxit: Iteration cap of the BiCGStab solve.
        poisson_solver: ``"direct"`` (default, exact) or ``"bicgstab"`` (float64 only; see the module docstring).
        preconditioner: ``"none"`` or ``"diag"`` (right preconditioning of the BiCGStab solve).
        dtype: ``"float32"`` (default) or ``"float64"`` (needs ``jax_enable_x64``).
    """

    nx: int = 72
    ny: int = 72
    nz: int = 72
    lx: float = 2.0 * math.pi
    ly: float = math.pi
    lz: float = 2.0
    nu: float = 1.9e-3
    re_tau: float = 180.0
    body_force: Optional[float] = None
    ctanh: float = 5e-2
    cfl: float = 0.9
    courant: str = "dnslab"
    rk_order: int = 4
    pressure_tol: float = 1e-3
    pressure_maxit: int = 2000
    poisson_solver: str = "direct"
    preconditioner: str = "none"
    dtype: str = "float32"


class ChannelFDState(NamedTuple):
    """Solver state: velocity components, the last pressure (warm start), the last time step and the time."""

    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    p: jnp.ndarray
    dt: jnp.ndarray
    t: jnp.ndarray


def butcher_tableau(order: int):
    """Explicit Runge-Kutta tableau ``(a, b)`` of the given order (1: Euler, 2: Heun, 3: Kutta, 4: classical RK4)."""
    if order == 1:
        return np.zeros((1, 1)), np.array([1.0])
    if order == 2:
        a = np.zeros((2, 2))
        a[1, 0] = 1.0
        return a, np.array([0.5, 0.5])
    if order == 3:
        a = np.zeros((3, 3))
        a[1, 0] = 0.5
        a[2, 0] = -1.0
        a[2, 1] = 2.0
        return a, np.array([1 / 6, 2 / 3, 1 / 6])
    if order == 4:
        a = np.zeros((4, 4))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.0
        return a, np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    raise ValueError(f"rk_order must be 1, 2, 3 or 4, got {order}")


def tanh_wall_grid(nz: int, height: float, ctanh: float):
    """Wall-normal faces, cell centres and cell widths of the ``tanh``-stretched grid.

    Returns:
        ``(zf, zc, dz)``: ``nz + 1`` face coordinates, ``nz`` cell-centre coordinates and ``nz`` cell widths
        (all float64 numpy arrays), identical to the grid of the reference code (``N3 = nz + 2``).
    """
    n3 = nz + 2
    k = np.arange(-(n3 // 2 - 1), n3 // 2)  # n3 - 1 faces
    fz = np.tanh(ctanh * k)
    fz = fz - fz[0]
    zc = 0.5 * (fz[:-1] + fz[1:])
    scale = height / fz[-1]
    return fz * scale, zc * scale, np.diff(fz) * scale


#######################################################################################
#                                                                                     #
#                                      SOLVER                                         #
#                                                                                     #
#######################################################################################


class ChannelFD:
    """Matrix-free finite-difference channel solver (pure functions of ``(u, v, w, p, dt)``).

    All derivative methods take and return arrays of shape ``(nx, ny, nz)``. ``dz``/``dzz`` accept the
    wall-normal values at the bottom/top wall (``bot``/``top``, shape ``(nx, ny)``, default 0) so that a
    transpiration velocity can be imposed on ``w``.
    """

    def __init__(self, config: ChannelFDConfig = ChannelFDConfig()):
        cfg = config
        if cfg.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {cfg.dtype!r}")
        if cfg.dtype == "float64" and not jax.config.jax_enable_x64:
            raise RuntimeError('dtype="float64" needs jax.config.update("jax_enable_x64", True) before the solver is built')
        if cfg.poisson_solver not in ("bicgstab", "direct"):
            raise ValueError(f"poisson_solver must be 'bicgstab' or 'direct', got {cfg.poisson_solver!r}")
        if cfg.poisson_solver == "bicgstab" and cfg.dtype != "float64":
            raise ValueError(
                'poisson_solver="bicgstab" needs dtype="float64" (the iteration is not reliable in 32-bit precision); '
                'use poisson_solver="direct" for float32'
            )
        if cfg.preconditioner not in ("none", "diag"):
            raise ValueError(f"preconditioner must be 'none' or 'diag', got {cfg.preconditioner!r}")
        if cfg.courant not in ("dnslab", "abs"):
            raise ValueError(f"courant must be 'dnslab' or 'abs', got {cfg.courant!r}")
        if cfg.nx < 3 or cfg.ny < 3 or cfg.nz < 3:
            raise ValueError("at least 3 cells are needed in every direction")
        self.cfg = cfg
        self.dtype = jnp.float64 if cfg.dtype == "float64" else jnp.float32
        self.shape = (cfg.nx, cfg.ny, cfg.nz)
        self.nu = cfg.nu
        self.gx = float(cfg.body_force) if cfg.body_force is not None else 8.0 * cfg.nu**2 * cfg.re_tau**2 / cfg.lz**3

        # --- grid (float64 numpy, then cast) ---
        self.dx_cell = cfg.lx / cfg.nx
        self.dy_cell = cfg.ly / cfg.ny
        zf, zc, dzc = tanh_wall_grid(cfg.nz, cfg.lz, cfg.ctanh)
        self.zf, self.zc, self.dz_cell = zf, zc, dzc
        self.x = self.dx_cell * (np.arange(cfg.nx) + 0.5)
        self.y = self.dy_cell * (np.arange(cfg.ny) + 0.5)

        # --- wall-normal stencil coefficients, ghost cells of the neighbour's width ---
        f0 = dzc
        fa = np.concatenate([dzc[1:], dzc[-1:]])  # width of the cell above (ghost = last cell)
        fg = np.concatenate([dzc[:1], dzc[:-1]])  # width of the cell below (ghost = first cell)
        cast = lambda a: jnp.asarray(a, dtype=self.dtype)  # noqa: E731
        # first derivative
        self._z1_c0 = cast(1.0 / f0 * (fa / (f0 + fa) - fg / (fg + f0)))
        self._z1_ca = cast(1.0 / (f0 + fa))
        self._z1_cg = cast(-1.0 / (f0 + fg))
        # second derivative
        ca2 = 2.0 / (f0 * (f0 + fa))
        cg2 = 2.0 / (f0 * (f0 + fg))
        self._z2_ca = cast(ca2)
        self._z2_cg = cast(cg2)
        self._z2_c0 = cast(-ca2 - cg2)
        # horizontal
        self._cx1 = 1.0 / (self.dx_cell + self.dx_cell)
        self._cy1 = 1.0 / (self.dy_cell + self.dy_cell)
        self._cx2 = 2.0 / (self.dx_cell * (self.dx_cell + self.dx_cell))
        self._cy2 = 2.0 / (self.dy_cell * (self.dy_cell + self.dy_cell))
        self._lap_xy_diag = -2.0 * self._cx2 - 2.0 * self._cy2
        # pressure pin: bottom-wall face below the cell (nx/2, round(ny/2)) of the reference grid
        self.pin_index = (int(round(cfg.nx / 2)) - 1, int(round(cfg.ny / 2)) - 1, 0)
        self._pin_coeff = float(-2.0 * cg2[0])  # extra diagonal when the ghost value is -p_0 instead of +p_0
        mask = np.zeros(self.shape)
        mask[self.pin_index] = 1.0
        self._pin_mask = cast(mask * self._pin_coeff)
        # inverse diagonal of the Poisson matrix (for the optional Jacobi preconditioner)
        zdiag = np.asarray(self._z2_c0) + np.concatenate([[cg2[0]], np.zeros(cfg.nz - 2), [ca2[-1]]])
        diag = self._lap_xy_diag + zdiag[None, None, :] + mask * self._pin_coeff
        self._inv_diag = cast(1.0 / diag)

        self.rk_a, self.rk_b = butcher_tableau(cfg.rk_order)
        self._dz_c = cast(dzc)

        self._direct = None  # tables of the direct solver (built here for "direct", else on first use)
        if cfg.poisson_solver == "direct":
            self._direct = _DirectPoisson(self)
        self.pressure_solve: Callable = self._solve_bicgstab if cfg.poisson_solver == "bicgstab" else self._solve_direct

    # ------------------------------------------------------------------
    # coordinates and geometry helpers
    # ------------------------------------------------------------------
    def grid(self):
        """Cell-centre coordinates ``(x, y, z)`` (float64 numpy, 1D)."""
        return self.x, self.y, self.zc

    # ------------------------------------------------------------------
    # horizontal operators (periodic, uniform)
    # ------------------------------------------------------------------
    def dx(self, f):
        """Streamwise first derivative (central, periodic)."""
        return (jnp.roll(f, -1, 0) - jnp.roll(f, 1, 0)) * self._cx1

    def dy(self, f):
        """Spanwise first derivative (central, periodic)."""
        return (jnp.roll(f, -1, 1) - jnp.roll(f, 1, 1)) * self._cy1

    dxp = dx  # periodic: the pressure and velocity operators coincide
    dyp = dy

    def dxx(self, f):
        """Streamwise second derivative."""
        return (jnp.roll(f, -1, 0) + jnp.roll(f, 1, 0) - 2.0 * f) * self._cx2

    def dyy(self, f):
        """Spanwise second derivative."""
        return (jnp.roll(f, -1, 1) + jnp.roll(f, 1, 1) - 2.0 * f) * self._cy2

    # ------------------------------------------------------------------
    # wall-normal operators
    # ------------------------------------------------------------------
    def _neighbours_z(self, f, ghost_lo, ghost_hi):
        below = jnp.concatenate([ghost_lo[..., None], f[..., :-1]], axis=-1)
        above = jnp.concatenate([f[..., 1:], ghost_hi[..., None]], axis=-1)
        return below, above

    def _dirichlet_ghosts(self, f, bot, top):
        g_lo = -f[..., 0] if bot is None else 2.0 * bot - f[..., 0]
        g_hi = -f[..., -1] if top is None else 2.0 * top - f[..., -1]
        return g_lo, g_hi

    def dz(self, f, bot=None, top=None):
        """Wall-normal first derivative with the Dirichlet ghost value ``2 b - f_0`` (no-slip: ``b = 0``)."""
        g_lo, g_hi = self._dirichlet_ghosts(f, bot, top)
        below, above = self._neighbours_z(f, g_lo, g_hi)
        return self._z1_c0 * f + self._z1_ca * above + self._z1_cg * below

    def dzp(self, f):
        """Wall-normal first derivative with a zero-gradient ghost value (pressure and flux derivatives)."""
        below, above = self._neighbours_z(f, f[..., 0], f[..., -1])
        return self._z1_c0 * f + self._z1_ca * above + self._z1_cg * below

    def dzz(self, f, bot=None, top=None):
        """Wall-normal second derivative with the Dirichlet ghost value ``2 b - f_0``."""
        g_lo, g_hi = self._dirichlet_ghosts(f, bot, top)
        below, above = self._neighbours_z(f, g_lo, g_hi)
        return self._z2_c0 * f + self._z2_ca * above + self._z2_cg * below

    # ------------------------------------------------------------------
    # Poisson operator
    # ------------------------------------------------------------------
    def poisson(self, p):
        """Poisson operator ``M p`` (zero normal gradient at the walls, ``p = 0`` on one bottom-wall face)."""
        lap_xy = (
            (jnp.roll(p, -1, 0) + jnp.roll(p, 1, 0)) * self._cx2
            + (jnp.roll(p, -1, 1) + jnp.roll(p, 1, 1)) * self._cy2
            + self._lap_xy_diag * p
        )
        below, above = self._neighbours_z(p, p[..., 0], p[..., -1])
        return lap_xy + self._z2_c0 * p + self._z2_ca * above + self._z2_cg * below + self._pin_mask * p

    # ------------------------------------------------------------------
    # pressure solves
    # ------------------------------------------------------------------
    def _solve_bicgstab(self, rhs, p0):
        """Solve ``M p = rhs`` (warm start ``p0``). Returns ``(p, num_steps)``."""
        # the stopping tolerance and the warm start do not influence the solution: no gradient flows through them
        atol = lax.stop_gradient(jnp.maximum(self.cfg.pressure_tol * jnp.linalg.norm(rhs.ravel()), _EPS_TINY))
        p0 = lax.stop_gradient(p0)
        maxit = self.cfg.pressure_maxit
        inv_diag = self._inv_diag
        if _lx is None:
            from jax.scipy.sparse.linalg import bicgstab

            kwargs = dict(x0=p0, tol=self.cfg.pressure_tol, atol=0.0, maxiter=maxit)
            if self.cfg.preconditioner == "diag":
                kwargs["M"] = lambda x: inv_diag * x
            p, _ = bicgstab(self.poisson, rhs, **kwargs)
            p = jnp.where(jnp.any(rhs != 0), p, jnp.zeros_like(p))
            return p, jnp.asarray(-1, jnp.int32)
        struct = jax.ShapeDtypeStruct(self.shape, self.dtype)
        op = _lx.FunctionLinearOperator(self.poisson, struct)
        solver = _BiCGStab(rtol=0.0, atol=atol, norm=_two_norm, max_steps=maxit)
        options = {"y0": p0}
        if self.cfg.preconditioner == "diag":
            options["preconditioner"] = _lx.FunctionLinearOperator(lambda x: inv_diag * x, struct)
        sol = _lx.linear_solve(op, rhs, solver, options=options, throw=False)
        return sol.value, sol.stats["num_steps"]

    def _solve_direct(self, rhs, p0):
        """Direct solve of ``M p = rhs`` (Fourier in x and y, tridiagonal in z); ignores ``p0``."""
        if self._direct is None:
            with jax.ensure_compile_time_eval():  # constants, never tracers, even when first called under jit/grad
                self._direct = _DirectPoisson(self)
        return self._direct.solve(rhs), jnp.asarray(0, jnp.int32)

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------
    def divergence(self, u, v, w, w_bot=None, w_top=None):
        """Discrete divergence ``dx u + dy v + dz w`` (``w_bot``/``w_top``: wall values of ``w``)."""
        return self.dx(u) + self.dy(v) + self.dz(w, w_bot, w_top)

    def project(self, u, v, w, p0, w_bot=None, w_top=None):
        """Pressure projection: returns ``(u, v, w, p, num_steps)`` with ``u <- u - dxp p`` etc."""
        div = self.divergence(u, v, w, w_bot, w_top)
        p, nit = self.pressure_solve(div, p0)
        return u - self.dxp(p), v - self.dyp(p), w - self.dzp(p), p, nit

    # ------------------------------------------------------------------
    # right-hand side and time stepping
    # ------------------------------------------------------------------
    def rhs(self, u, v, w, w_bot=None, w_top=None):
        """Momentum right-hand side ``-conv + nu * lap + g`` for ``(u, v, w)``; returns ``(ku, kv, kw)``."""
        nu = self.nu

        def comp(q, wall_bot=None, wall_top=None):
            dq_dz = self.dz(q, wall_bot, wall_top)
            conv = 0.5 * (
                self.dx(u * q) + self.dy(v * q) + self.dzp(w * q) + u * self.dx(q) + v * self.dy(q) + w * dq_dz
            )
            diff = nu * (self.dxx(q) + self.dyy(q) + self.dzz(q, wall_bot, wall_top))
            return diff - conv

        ku = comp(u) + self.gx
        kv = comp(v)
        kw = comp(w, w_bot, w_top)
        return ku, kv, kw

    def courant_number(self, u, v, w, dt):
        """Maximum Courant number of the field for the step ``dt`` (DNSLab: signed sum of the three directions)."""
        if self.cfg.courant == "abs":
            u, v, w = jnp.abs(u), jnp.abs(v), jnp.abs(w)
        co = u * dt / self.dx_cell + v * dt / self.dy_cell + w * dt / self._dz_c
        return jnp.max(co)

    def adjust_dt(self, u, v, w, dt):
        """Time step for the target Courant number: ``dt / (Co_max / cfl)``."""
        return dt / (self.courant_number(u, v, w, dt) / self.cfg.cfl)

    def step_dt(self, u, v, w, p, dt, w_bot=None, w_top=None):
        """One Runge-Kutta step of size ``dt`` with a projection after every stage.

        Returns ``(u, v, w, p, nits)``; ``nits`` is the vector of the Poisson iteration counts of the stages
        (``rk_order`` stage projections after the first stage, plus the final one).
        """
        a, b = self.rk_a, self.rk_b
        s = len(b)
        uo, vo, wo = u, v, w
        uc, vc, wc = u, v, w
        ks = []
        nits = []
        for ii in range(s):
            if ii >= 1:
                du = sum(float(a[ii, j]) * ks[j][0] for j in range(ii) if a[ii, j] != 0.0)
                dv = sum(float(a[ii, j]) * ks[j][1] for j in range(ii) if a[ii, j] != 0.0)
                dw = sum(float(a[ii, j]) * ks[j][2] for j in range(ii) if a[ii, j] != 0.0)
                u, v, w, p, nit = self.project(uo + dt * du, vo + dt * dv, wo + dt * dw, p, w_bot, w_top)
                nits.append(nit)
            k = self.rhs(u, v, w, w_bot, w_top)
            ks.append(k)
            uc = uc + dt * float(b[ii]) * k[0]
            vc = vc + dt * float(b[ii]) * k[1]
            wc = wc + dt * float(b[ii]) * k[2]
        u, v, w, p, nit = self.project(uc, vc, wc, p, w_bot, w_top)
        nits.append(nit)
        return u, v, w, p, jnp.stack(nits)

    def init_state(self, u, v, w, project: bool = True, w_bot=None, w_top=None, dt0: Optional[float] = None):
        """Build a state from velocity fields (optionally projected first, as the reference code does)."""
        u = jnp.asarray(u, self.dtype)
        v = jnp.asarray(v, self.dtype)
        w = jnp.asarray(w, self.dtype)
        p = jnp.zeros(self.shape, self.dtype)
        if project:
            u, v, w, p, _ = self.project(u, v, w, p, w_bot, w_top)
        if dt0 is None:
            dt0 = self.cfg.cfl * self.dx_cell / 7.182  # DNSLab's initial guess; rescaled by the first Courant check
        return ChannelFDState(u, v, w, p, jnp.asarray(dt0, self.dtype), jnp.asarray(0.0, self.dtype))

    def step(self, state: ChannelFDState, w_bot=None, w_top=None, dt=None):
        """Advance one step with the dynamic time step (or the fixed ``dt``).

        Returns ``(new_state, aux)`` with ``aux = {"dt", "courant", "nits"}``.
        """
        u, v, w, p, dt_prev, t = state
        if dt is None:
            dt = self.adjust_dt(u, v, w, dt_prev)
        else:
            dt = jnp.asarray(dt, self.dtype)
        u, v, w, p, nits = self.step_dt(u, v, w, p, dt, w_bot, w_top)
        co = self.courant_number(u, v, w, dt)
        return ChannelFDState(u, v, w, p, dt, t + dt), {"dt": dt, "courant": co, "nits": nits}

    def run(self, state: ChannelFDState, n_steps: int, w_bot=None, w_top=None, dt=None, record: Optional[Callable] = None):
        """``n_steps`` steps with ``lax.scan``; ``record(state, aux)`` (optional) gives the per-step outputs."""

        def body(s, _):
            s, aux = self.step(s, w_bot, w_top, dt)
            return s, (record(s, aux) if record is not None else aux)

        return lax.scan(body, state, None, length=n_steps)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def wall_shear(self, u, which: str = "mean"):
        """Wall shear stress ``nu * u / z`` from the first cell at the bottom and the top wall.

        Args:
            u: Streamwise velocity ``(nx, ny, nz)``.
            which: ``"bottom"``, ``"top"`` or ``"mean"`` (mean of both walls). Returns the plane mean.
        """
        tb = jnp.mean(self.nu * u[..., 0] / float(self.zc[0]))
        tt = jnp.mean(self.nu * u[..., -1] / float(self.cfg.lz - self.zc[-1]))
        return {"bottom": tb, "top": tt, "mean": 0.5 * (tb + tt)}[which]

    def bulk_velocity(self, u):
        """Volume-averaged streamwise velocity (cell-width weighted)."""
        w = self._dz_c / self.cfg.lz
        return jnp.mean(jnp.sum(u * w, axis=-1))


#######################################################################################
#                                                                                     #
#                           DIRECT (FFT + TRIDIAGONAL) SOLVER                         #
#                                                                                     #
#######################################################################################


class _DirectPoisson:
    """Exact solver of ``M p = rhs`` for the Poisson operator of ``ChannelFD`` (pinned pressure).

    ``M = M0 + c e e^T`` where ``M0`` (zero normal gradient everywhere) is separable and singular (its null space
    is the constants) and ``e`` selects the pinned cell. Fourier modes in x and y turn ``M0`` into independent
    tridiagonal systems in z. The pinned value follows from the compatibility condition of ``M0`` (its left null
    vector ``l`` is constant in x, y): ``l^T rhs = c p_e l_e``. The remaining compatible singular system is solved
    with the additive constant fixed so that ``p`` at the pinned cell equals ``p_e``.
    """

    def __init__(self, solver: ChannelFD):
        self.s = solver
        s = solver
        nx, ny, nz = s.shape
        kx = np.arange(nx)
        ky = np.arange(ny)
        lam_x = -2.0 * s._cx2 * (1.0 - np.cos(2.0 * np.pi * kx / nx))
        lam_y = -2.0 * s._cy2 * (1.0 - np.cos(2.0 * np.pi * ky / ny))
        lam = lam_x[:, None] + lam_y[None, :]  # eigenvalues of the horizontal Laplacian, (nx, ny)
        c0 = np.asarray(s._z2_c0, dtype=np.float64).copy()
        ca = np.asarray(s._z2_ca, dtype=np.float64)
        cg = np.asarray(s._z2_cg, dtype=np.float64)
        c0[0] += cg[0]  # zero-gradient walls: ghost value = +p_0
        c0[-1] += ca[-1]
        lam_nz = lam.copy()
        lam_nz[0, 0] = 1.0  # the singular mode is handled separately; keep the batch solve finite
        self.lam = jnp.asarray(lam_nz, s.dtype)
        self.lower = jnp.asarray(cg, s.dtype)  # coefficient of p_{k-1} in row k
        self.upper = jnp.asarray(ca, s.dtype)  # coefficient of p_{k+1} in row k
        self.diag0 = jnp.asarray(c0, s.dtype)
        T0 = np.diag(c0) + np.diag(ca[:-1], 1) + np.diag(cg[1:], -1)
        # left null vector l (l^T T0 = 0): right null vector of T0^T
        lvec = np.linalg.svd(T0.T)[2][-1]
        lvec = lvec / lvec[np.argmax(np.abs(lvec))]
        self.lvec = jnp.asarray(lvec, s.dtype)
        # zero horizontal mode: T0 x = r is compatible; replace row 0 by x_0 = 0 (l_0 != 0 makes it redundant)
        Tz = T0.copy()
        Tz[0, :] = 0.0
        Tz[0, 0] = 1.0
        self.tz_inv = jnp.asarray(np.linalg.inv(Tz), s.dtype)
        self.pin = s.pin_index
        self.cpin = float(s._pin_coeff)
        self.l_pin = float(lvec[self.pin[2]])

    def _tridiag(self, d, b):
        """Thomas algorithm along the last axis for every horizontal mode (``d``: diagonals, ``b``: right sides)."""
        lo, up = self.lower, self.upper
        nz = lo.shape[0]

        def fwd(carry, k):
            cprime, dprime = carry
            denom = d[..., k] - lo[k] * cprime
            cp = up[k] / denom
            dp = (b[..., k] - lo[k] * dprime) / denom
            return (cp, dp), (cp, dp)

        zero = jnp.zeros_like(b[..., 0])
        _, (cps, dps) = lax.scan(fwd, (zero, zero), jnp.arange(nz))

        def bwd(x_next, k):
            x = dps[k] - cps[k] * x_next
            return x, x

        _, xs = lax.scan(bwd, zero, jnp.arange(nz - 1, -1, -1))
        return jnp.moveaxis(xs[::-1], 0, -1)

    def solve(self, rhs):
        # pinned value from the compatibility condition l^T sum_xy(rhs) = c p_e l_e
        p_e = jnp.sum(self.lvec * jnp.sum(rhs, axis=(0, 1))) / (self.cpin * self.l_pin)
        r = rhs.at[self.pin].add(-self.cpin * p_e)
        rhat = jnp.fft.fft2(r, axes=(0, 1))
        d = self.diag0[None, None, :] + self.lam[:, :, None]
        sol = self._tridiag(d, rhat)
        zero_mode = self.tz_inv @ rhat[0, 0, :].at[0].set(0.0)  # replaced row 0: x_0 = 0
        sol = sol.at[0, 0, :].set(zero_mode)
        p = jnp.fft.ifft2(sol, axes=(0, 1)).real.astype(rhs.dtype)
        return p + (p_e - p[self.pin])
