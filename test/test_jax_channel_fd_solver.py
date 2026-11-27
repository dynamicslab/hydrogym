"""Unit tests for hydrogym.jax.solvers.channel_fd (finite-difference channel solver).

Small grids only; float64 (x64 is enabled here for the whole module). Operators are checked against dense matrices
assembled with explicit loops from the stencil definitions, the pressure solve against its residual and a direct
solve, the time integration against Poiseuille flow and the exact solution of a linear problem, and the gradients
against central finite differences.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from hydrogym.jax.solvers.channel_fd import ChannelFD, ChannelFDConfig, butcher_tableau, tanh_wall_grid  # noqa: E402

NX, NY, NZ = 6, 5, 8


def _cfg(**kw):
    """Configuration with the float64 BiCGStab defaults (the package default is float32 with the direct solver)."""
    return ChannelFDConfig(**{"dtype": "float64", "poisson_solver": "bicgstab", **kw})


def small_solver(**kw):
    kw = {"nx": NX, "ny": NY, "nz": NZ, "ctanh": 0.15, **kw}
    return ChannelFD(_cfg(**kw))


def assemble_reference(s):
    """Dense matrices of the ten operators, assembled by explicit loops (index (i * ny + j) * nz + k)."""
    nx, ny, nz = s.shape
    N = nx * ny * nz
    dx, dy = s.dx_cell, s.dy_cell
    F = np.concatenate([[s.dz_cell[0]], s.dz_cell, [s.dz_cell[-1]]])  # widths with ghost cells
    ix = lambda i, j, k: ((i % nx) * ny + (j % ny)) * nz + k  # noqa: E731
    names = ["Dx", "Dy", "Dz", "Dxx", "Dyy", "Dzz", "Dxp", "Dyp", "Dzp", "M"]
    A = {n: np.zeros((N, N)) for n in names}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = ix(i, j, k)
                f0, fa, fg = F[k + 1], F[k + 2], F[k]
                for n in ("Dx", "Dxp"):
                    A[n][r, ix(i + 1, j, k)] += 1 / (dx + dx)
                    A[n][r, ix(i - 1, j, k)] -= 1 / (dx + dx)
                for n in ("Dy", "Dyp"):
                    A[n][r, ix(i, j + 1, k)] += 1 / (dy + dy)
                    A[n][r, ix(i, j - 1, k)] -= 1 / (dy + dy)
                A["Dxx"][r, ix(i + 1, j, k)] += 2 / (dx * 2 * dx)
                A["Dxx"][r, ix(i - 1, j, k)] += 2 / (dx * 2 * dx)
                A["Dxx"][r, r] -= 4 / (dx * 2 * dx)
                A["Dyy"][r, ix(i, j + 1, k)] += 2 / (dy * 2 * dy)
                A["Dyy"][r, ix(i, j - 1, k)] += 2 / (dy * 2 * dy)
                A["Dyy"][r, r] -= 4 / (dy * 2 * dy)
                # wall-normal first derivative: coefficients of the cell above, below and the cell itself
                ca, cg = 1 / (f0 + fa), -1 / (f0 + fg)
                c0 = (fa / (f0 + fa) - fg / (fg + f0)) / f0
                for n, sign in (("Dz", -1.0), ("Dzp", +1.0)):  # ghost value = sign * u_0
                    A[n][r, r] += c0
                    if k + 1 < nz:
                        A[n][r, ix(i, j, k + 1)] += ca
                    else:
                        A[n][r, r] += sign * ca
                    if k > 0:
                        A[n][r, ix(i, j, k - 1)] += cg
                    else:
                        A[n][r, r] += sign * cg
                # second derivative
                ca2, cg2 = 2 / (f0 * (f0 + fa)), 2 / (f0 * (f0 + fg))
                A["Dzz"][r, r] -= ca2 + cg2
                if k + 1 < nz:
                    A["Dzz"][r, ix(i, j, k + 1)] += ca2
                else:
                    A["Dzz"][r, r] -= ca2  # ghost value -u_0
                if k > 0:
                    A["Dzz"][r, ix(i, j, k - 1)] += cg2
                else:
                    A["Dzz"][r, r] -= cg2
                # Poisson operator with zero normal gradient at the walls
                cx, cy = 2 / (dx * 2 * dx), 2 / (dy * 2 * dy)
                A["M"][r, ix(i + 1, j, k)] += cx
                A["M"][r, ix(i - 1, j, k)] += cx
                A["M"][r, ix(i, j + 1, k)] += cy
                A["M"][r, ix(i, j - 1, k)] += cy
                A["M"][r, r] -= 2 * cx + 2 * cy
                if k + 1 < nz:
                    A["M"][r, ix(i, j, k + 1)] += ca2
                    A["M"][r, r] -= ca2
                if k > 0:
                    A["M"][r, ix(i, j, k - 1)] += cg2
                    A["M"][r, r] -= cg2
    # pressure fixed to zero on one bottom-wall face: ghost value -p_0 instead of +p_0 in that cell
    i0, j0, k0 = s.pin_index
    f0, fg = F[1], F[0]
    r = ix(i0, j0, k0)
    A["M"][r, r] += -4 / (f0 * (f0 + fg))
    return A


def test_operators_match_assembled_matrices():
    s = small_solver()
    A = assemble_reference(s)
    rng = np.random.default_rng(1)
    ops = {
        "Dx": s.dx,
        "Dy": s.dy,
        "Dz": s.dz,
        "Dxx": s.dxx,
        "Dyy": s.dyy,
        "Dzz": s.dzz,
        "Dxp": s.dxp,
        "Dyp": s.dyp,
        "Dzp": s.dzp,
        "M": s.poisson,
    }
    for _ in range(2):
        f = rng.standard_normal(s.shape)
        for n, fn in ops.items():
            ref = A[n] @ f.ravel()
            got = np.asarray(fn(jnp.asarray(f))).ravel()
            assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max(), n


def test_pin_cell_and_null_space():
    """Without the pin the Poisson operator annihilates constants; with it the operator is nonsingular."""
    s = small_solver()
    one = jnp.ones(s.shape)
    out = np.asarray(s.poisson(one))
    pin = s.pin_index
    mask = np.zeros(s.shape, bool)
    mask[pin] = True
    assert np.abs(out[~mask]).max() < 1e-10
    assert abs(out[pin]) > 1.0


def test_wall_values_enter_the_ghost_cells():
    """A transpiration value b at the bottom wall adds 2 b times the below-coefficient to the first derivative."""
    s = small_solver()
    f = jnp.asarray(np.random.default_rng(2).standard_normal(s.shape))
    b = jnp.asarray(np.random.default_rng(3).standard_normal(s.shape[:2]))
    d0, d1 = s.dz(f), s.dz(f, bot=b)
    diff = np.asarray(d1 - d0)
    assert np.abs(diff[..., 1:]).max() == 0.0
    np.testing.assert_allclose(diff[..., 0], 2.0 * np.asarray(b) * float(s._z1_cg[0]), rtol=1e-12)
    dd = np.asarray(s.dzz(f, bot=b) - s.dzz(f))
    np.testing.assert_allclose(dd[..., 0], 2.0 * np.asarray(b) * float(s._z2_cg[0]), rtol=1e-12)


def test_grid_is_symmetric_and_spans_the_height():
    zf, zc, dz = tanh_wall_grid(72, 2.0, 5e-2)
    assert zf[0] == pytest.approx(0.0) and zf[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(dz, dz[::-1], rtol=1e-12)
    np.testing.assert_allclose(zc + zc[::-1], 2.0, rtol=1e-12)
    assert dz.min() < dz.max()


def test_butcher_tableaus_are_consistent():
    for order in (1, 2, 3, 4):
        a, b = butcher_tableau(order)
        assert b.sum() == pytest.approx(1.0)
        assert np.allclose(np.triu(a), 0.0)
    with pytest.raises(ValueError):
        butcher_tableau(5)


def test_pressure_solve_residual_and_direct_agreement():
    s = ChannelFD(_cfg(nx=16, ny=16, nz=16, ctanh=0.1, pressure_tol=1e-10, pressure_maxit=5000))
    sd = ChannelFD(_cfg(nx=16, ny=16, nz=16, ctanh=0.1, poisson_solver="direct"))
    rng = np.random.default_rng(4)
    rhs = jnp.asarray(rng.standard_normal(s.shape))
    p, nit = s.pressure_solve(rhs, jnp.zeros(s.shape))
    assert int(nit) > 0
    assert float(jnp.linalg.norm((s.poisson(p) - rhs).ravel()) / jnp.linalg.norm(rhs.ravel())) < 2e-10
    pd, _ = sd.pressure_solve(rhs, jnp.zeros(s.shape))
    assert float(jnp.linalg.norm((sd.poisson(pd) - rhs).ravel()) / jnp.linalg.norm(rhs.ravel())) < 1e-11
    assert float(jnp.linalg.norm((pd - p).ravel()) / jnp.linalg.norm(pd.ravel())) < 1e-6


def test_pressure_solve_tolerance_and_zero_rhs():
    s = ChannelFD(_cfg(nx=16, ny=16, nz=16, ctanh=0.1, pressure_tol=1e-3))
    rhs = jnp.asarray(np.random.default_rng(5).standard_normal(s.shape))
    p, _ = s.pressure_solve(rhs, jnp.zeros(s.shape))
    assert float(jnp.linalg.norm((s.poisson(p) - rhs).ravel()) / jnp.linalg.norm(rhs.ravel())) <= 1e-3
    for solver in ("bicgstab", "direct"):
        sz = ChannelFD(_cfg(nx=8, ny=8, nz=8, poisson_solver=solver))
        pz, _ = sz.pressure_solve(jnp.zeros(sz.shape), jnp.zeros(sz.shape))
        assert np.all(np.asarray(pz) == 0.0)


def test_projection_removes_the_divergence_of_a_smooth_field():
    s = ChannelFD(_cfg(nx=24, ny=24, nz=24, ctanh=0.1, pressure_tol=1e-8, pressure_maxit=5000))
    x, y, z = (jnp.asarray(a) for a in s.grid())
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    env = jnp.sin(jnp.pi * Z / 2.0)
    u = env * jnp.cos(X) * jnp.cos(2 * Y)
    v = env * jnp.sin(2 * Y) * jnp.cos(X)
    w = env**2 * jnp.sin(X) * jnp.cos(2 * Y)
    d0 = s.divergence(u, v, w)
    u1, v1, w1, p, _ = s.project(u, v, w, jnp.zeros(s.shape))
    d1 = s.divergence(u1, v1, w1)
    assert float(jnp.sqrt(jnp.mean(d1**2)) / jnp.sqrt(jnp.mean(d0**2))) < 0.05


def test_dynamic_time_step_hits_the_target_courant_number():
    s = small_solver(cfl=0.7)
    rng = np.random.default_rng(6)
    u = jnp.asarray(1.0 + 0.2 * rng.standard_normal(s.shape))
    v = jnp.asarray(0.1 * rng.standard_normal(s.shape))
    w = jnp.asarray(0.1 * rng.standard_normal(s.shape))
    dt = s.adjust_dt(u, v, w, jnp.asarray(0.01))
    assert float(s.courant_number(u, v, w, dt)) == pytest.approx(0.7, rel=1e-12)
    sa = small_solver(cfl=0.7, courant="abs")
    dta = sa.adjust_dt(u, v, w, jnp.asarray(0.01))
    assert float(sa.courant_number(u, v, w, dta)) == pytest.approx(0.7, rel=1e-12)
    assert float(dta) <= float(dt) + 1e-15  # absolute values give the larger Courant number


def test_laminar_poiseuille_flow_is_steady_to_second_order():
    """The discrete steady state of the forced channel converges to the parabola at second order; the flow stays put."""
    errs = []
    for nz in (12, 24, 48):
        s = ChannelFD(_cfg(nx=4, ny=4, nz=nz, ctanh=0.05 * 72 / nz, poisson_solver="direct"))
        A = np.stack(
            [np.asarray(s.dzz(jnp.broadcast_to(jnp.zeros(nz).at[j].set(1.0), s.shape))[0, 0]) for j in range(nz)],
            axis=1,
        )
        ue = s.gx / (2 * s.nu) * s.zc * (s.cfg.lz - s.zc)
        ud = np.linalg.solve(s.nu * A, -s.gx * np.ones(nz))
        errs.append(np.abs(ud - ue).max() / ue.max())
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(1.8 < o < 2.2 for o in orders), orders
    s = ChannelFD(_cfg(nx=8, ny=8, nz=24, ctanh=0.15, poisson_solver="direct"))
    u = jnp.broadcast_to(s.gx / (2 * s.nu) * jnp.asarray(s.zc) * (s.cfg.lz - jnp.asarray(s.zc)), s.shape)
    z0 = jnp.zeros(s.shape)
    st, _ = s.run(s.init_state(u, z0, z0), 100)
    # the flow relaxes towards the discrete steady state, which differs from the parabola by the discretisation error
    assert float(jnp.abs(st.u - u).max() / jnp.abs(u).max()) < 1.5 * errs[1]
    assert float(jnp.abs(st.v).max()) == 0.0 and float(jnp.abs(st.w).max()) == 0.0


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_runge_kutta_order_on_a_linear_problem(order):
    """x-independent diffusion with a body force (projection trivial): observed order equals the scheme order."""
    import scipy.linalg as sla

    nx = ny = 8
    nz = 12
    nu, T = 0.3, 0.2
    base = ChannelFD(_cfg(nx=nx, ny=ny, nz=nz, nu=nu, poisson_solver="direct", rk_order=order))
    npl = ny * nz
    eye = jnp.eye(npl).reshape(npl, ny, nz)
    lap = jax.vmap(
        lambda e: nu * (base.dyy(jnp.broadcast_to(e, base.shape)) + base.dzz(jnp.broadcast_to(e, base.shape)))[0]
    )(eye)
    A = np.asarray(lap).reshape(npl, npl).T
    aug = np.zeros((npl + 1, npl + 1))
    aug[:npl, :npl] = A
    aug[:npl, npl] = base.gx
    Y, Z = np.meshgrid(base.y, base.zc, indexing="ij")
    u0p = np.sin(np.pi * Z / 2.0) * np.cos(2 * np.pi * 2 * Y / base.cfg.ly)
    exact = (sla.expm(T * aug) @ np.concatenate([u0p.ravel(), [1.0]]))[:npl].reshape(ny, nz)
    zero = jnp.zeros(base.shape)
    st0 = base.init_state(jnp.broadcast_to(jnp.asarray(u0p)[None], base.shape), zero, zero)
    errs = []
    for n in (40, 80, 160):
        st, _ = base.run(st0, n, dt=T / n)
        errs.append(float(np.abs(np.asarray(st.u)[0] - exact).max()))
    obs = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert all(abs(o - order) < 0.15 for o in obs), (order, obs)


def test_step_is_independent_of_the_pressure_solver_at_tight_tolerance():
    kw = dict(nx=12, ny=12, nz=12, ctanh=0.1, cfl=0.5)
    sb = ChannelFD(_cfg(**kw, pressure_tol=1e-11, pressure_maxit=5000))
    sd = ChannelFD(_cfg(**kw, poisson_solver="direct"))
    x, y, z = (jnp.asarray(a) for a in sb.grid())
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    env = jnp.sin(jnp.pi * Z / 2.0)
    u0 = 3.0 * env * (1.0 + 0.2 * jnp.cos(X) * jnp.cos(2 * Y))
    v0 = 0.2 * env * jnp.sin(X)
    w0 = 0.1 * env**2 * jnp.cos(2 * Y)
    a, _ = sb.run(sb.init_state(u0, v0, w0), 5)
    b, _ = sd.run(sd.init_state(u0, v0, w0), 5)
    for f in ("u", "v", "w"):
        assert float(jnp.abs(getattr(a, f) - getattr(b, f)).max()) < 1e-8
    assert float(abs(a.dt - b.dt)) < 1e-10


def test_gradient_of_the_wall_shear_matches_finite_differences():
    """d(mean wall shear after 3 steps)/d(wall-normal jet velocity), reverse mode through the lineax solve."""
    s = ChannelFD(_cfg(nx=8, ny=8, nz=8, ctanh=0.1, pressure_tol=1e-11, pressure_maxit=5000, rk_order=2))
    x, y, z = (jnp.asarray(a) for a in s.grid())
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    env = jnp.sin(jnp.pi * Z / 2.0)
    st0 = s.init_state(
        3.0 * env * (1.0 + 0.2 * jnp.cos(X) * jnp.cos(2 * Y)), 0.2 * env * jnp.sin(X), 0.1 * env**2 * jnp.cos(2 * Y)
    )
    Xp, Yp = jnp.meshgrid(x, y, indexing="ij")
    shapes = [jnp.cos(Xp) * jnp.cos(Yp), jnp.sin(2 * Xp) * jnp.sin(Yp), jnp.cos(2 * Yp) * jnp.cos(Xp)]

    def wss(c):
        wall = 0.05 * sum(ci * sh for ci, sh in zip(c, shapes))
        st, _ = s.run(st0, 3, w_bot=wall, dt=0.01)
        return s.wall_shear(st.u, "bottom")

    c0 = jnp.asarray([0.3, -0.2, 0.5])
    g = np.asarray(jax.grad(wss)(c0))
    assert np.all(np.isfinite(g))
    eps = 1e-5
    fd = np.array([(wss(c0.at[i].add(eps)) - wss(c0.at[i].add(-eps))) / (2 * eps) for i in range(3)])
    assert np.linalg.norm(g - fd) <= 1e-3 * np.linalg.norm(fd)
    assert np.abs(fd).max() > 1e-8  # a non-trivial sensitivity


def test_gradient_through_the_direct_solver():
    s = ChannelFD(_cfg(nx=8, ny=8, nz=8, ctanh=0.1, poisson_solver="direct", rk_order=2))
    x, y, z = (jnp.asarray(a) for a in s.grid())
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    env = jnp.sin(jnp.pi * Z / 2.0)
    st0 = s.init_state(3.0 * env * (1.0 + 0.2 * jnp.cos(X)), 0.2 * env * jnp.sin(X), 0.1 * env**2 * jnp.cos(2 * Y))
    Xp, Yp = jnp.meshgrid(x, y, indexing="ij")

    def wss(c):
        st, _ = s.run(st0, 3, w_bot=0.05 * c * jnp.cos(Xp) * jnp.cos(Yp), dt=0.01)
        return s.wall_shear(st.u, "bottom")

    g = float(jax.grad(wss)(0.4))
    fd = float((wss(0.4 + 1e-5) - wss(0.4 - 1e-5)) / 2e-5)
    assert g == pytest.approx(fd, rel=1e-4)


def test_float32_step_runs_and_stays_finite():
    s = ChannelFD(_cfg(nx=12, ny=12, nz=12, ctanh=0.1, dtype="float32", poisson_solver="direct"))
    x, y, z = (jnp.asarray(a, jnp.float32) for a in s.grid())
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    env = jnp.sin(jnp.pi * Z / 2.0)
    st = s.init_state(3.0 * env * (1.0 + 0.2 * jnp.cos(X)), 0.2 * env * jnp.sin(X), 0.1 * env**2)
    st, _ = s.run(st, 3)
    assert st.u.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(st.u)))


def test_float32_pressure_solve_is_accurate_without_x64():
    """With 64-bit precision off (the default of a fresh process) the default pressure solve is accurate."""
    import subprocess
    import sys

    code = """
import numpy as np, jax, jax.numpy as jnp
assert not jax.config.jax_enable_x64
from hydrogym.jax.solvers.channel_fd import ChannelFD, ChannelFDConfig
s = ChannelFD(ChannelFDConfig(nx=16, ny=16, nz=16, ctanh=0.1, dtype="float32", pressure_tol=1e-3))
rng = np.random.default_rng(0)
rhs = jnp.asarray(rng.standard_normal(s.shape), jnp.float32)
p, _ = s.pressure_solve(rhs, jnp.zeros(s.shape, jnp.float32))
res = float(jnp.linalg.norm((s.poisson(p) - rhs).ravel()) / jnp.linalg.norm(rhs.ravel()))
pz, _ = s.pressure_solve(jnp.zeros(s.shape, jnp.float32), jnp.ones(s.shape, jnp.float32))
print("RESIDUAL", res, float(jnp.abs(pz).max()))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESIDUAL")]
    assert line, out.stderr[-800:]
    res, zmax = (float(x) for x in line[0].split()[1:])
    assert res <= 2e-3 and zmax == 0.0


def test_bicgstab_needs_float64_and_the_default_is_the_float32_direct_solver():
    cfg = ChannelFDConfig()
    assert cfg.dtype == "float32" and cfg.poisson_solver == "direct"
    with pytest.raises(ValueError, match="float64"):
        ChannelFD(ChannelFDConfig(nx=8, ny=8, nz=8, poisson_solver="bicgstab", dtype="float32"))
