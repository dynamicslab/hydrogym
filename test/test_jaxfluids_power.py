"""Injection power of the JAX-Fluids nozzle (hydrogym.power), solver-free checks.

The power function must use exactly the injected mass flux of the interface-flux function the solver applies,
and its exergy form must reduce to the incompressible transpiration power ``Q (p_t - p_inf)`` at low pressure
ratio.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxfluids")
pytest.importorskip("pyvista")

import jax.numpy as jnp  # noqa: E402

from hydrogym.jaxfluids.utils import nozzle as nz  # noqa: E402

P_INF, T_INF, GAMMA, R_GAS = 1e5, 300.0, 1.4, 287.14


def _setup(n_inj=2, spr=0.9, npr=4.6):
    geom = nz.InjectorGeometry(X=0.789, IW=0.002032, N=n_inj)
    ratios = nz.PressureRatios(NPR=npr, SPR=spr)
    sim = SimpleNamespace(domain_information=SimpleNamespace(dim=2, smallest_cell_size=1e-4))
    kw = dict(
        injector_geometry=geom,
        pressure_ratios=ratios,
        p_infty=P_INF,
        T_infty=T_INF,
        specific_heat_ratio=GAMMA,
        specific_gas_constant=R_GAS,
        sim_manager=sim,
    )
    flux_fn = nz.initialize_injector_flux_fn(**kw)
    power_fn = nz.initialize_injector_power_fn(**kw)
    params = nz._compute_injector_plane_params(geom)
    # interface cells along each injector slot, normal = injector normal (into the nozzle flow)
    pts, normals = [], []
    for i in range(n_inj):
        s = np.linspace(-0.0012, 0.0012, 25)
        pos, tan, nrm = (np.asarray(a[i]) for a in (params.positions, params.tangents, params.normals))
        pts.append(pos[:2, None] + tan[:2, None] * s[None])
        normals.append(np.repeat(nrm[:, None], s.size, axis=1))
    mesh_grid = jnp.asarray(np.concatenate(pts, axis=1))
    normal = jnp.asarray(np.concatenate(normals, axis=1))
    n = mesh_grid.shape[1]
    return flux_fn, power_fn, mesh_grid, normal, n


def _primitives(n, p_local):
    rho = p_local / (R_GAS * T_INF)
    return jnp.stack([jnp.full(n, rho), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), jnp.full(n, p_local)])


@pytest.mark.parametrize("p_local", [0.6e5, 1.4e5, 3.0e5])
def test_power_uses_the_solver_mass_flux(p_local):
    flux_fn, power_fn, mesh_grid, normal, n = _setup()
    length = jnp.full(n, 1e-4)
    prim = _primitives(n, p_local)
    action = jnp.array([1.0, 0.4])
    mass_flux, _, _ = flux_fn(prim, length, normal, mesh_grid, action)
    mdot, power = power_fn(prim, length, normal, mesh_grid, action)
    assert mdot.shape == power.shape == (2, n)
    np.testing.assert_allclose(np.asarray(jnp.sum(mdot, axis=0)), np.asarray(mass_flux), rtol=1e-12, atol=0)
    assert float(jnp.sum(mdot[0])) > 0  # injection into the flow
    # zero action: no injection, no power
    mdot0, power0 = power_fn(prim, length, normal, mesh_grid, jnp.zeros(2))
    assert float(jnp.max(jnp.abs(mdot0))) == 0.0 and float(jnp.max(jnp.abs(power0))) == 0.0


def test_exergy_power_low_mach_limit():
    """``mdot R T ln(p_t / p_inf) -> Q (p_t - p_inf)`` for ``p_t -> p_inf`` (Bewley's transpiration power)."""
    flux_fn, power_fn, mesh_grid, normal, n = _setup(spr=1.0, npr=1.0)  # P0 = p_inf
    length = jnp.full(n, 1e-4)
    p_local = 0.98 * P_INF
    action = jnp.array([1.0, 1.0])
    mdot, power = power_fn(_primitives(n, p_local), length, normal, mesh_grid, action)
    p_t = P_INF  # action 1: injector total pressure = reservoir pressure
    rho_t = p_t / (R_GAS * T_INF)
    q = float(jnp.sum(mdot)) / rho_t  # volume flow at reservoir density
    assert float(jnp.sum(mdot)) > 0 and float(jnp.sum(power)) == pytest.approx(0.0, abs=1e-9)  # p_t = p_inf: no exergy
    # slightly above ambient: exergy ~ Q (p_t - p_inf)
    flux_fn, power_fn, mesh_grid, normal, n = _setup(spr=1.0, npr=1.02)
    mdot, power = power_fn(_primitives(n, p_local), length, normal, mesh_grid, action)
    p_t = 1.02 * P_INF
    q = float(jnp.sum(mdot)) / (p_t / (R_GAS * T_INF))
    assert float(jnp.sum(power)) == pytest.approx(q * (p_t - P_INF), rel=2e-2)


def test_primary_reference_power():
    spec = nz.TVCSpec(dim=2, fixed_num_actuators=2)
    p_ref = nz.primary_reference_power(spec)
    mdot = 2 * spec.nozzle_geometry.R * spec.p0 * np.sqrt(GAMMA / (R_GAS * T_INF)) * (2 / 2.4) ** 3.0
    assert p_ref == pytest.approx(mdot * R_GAS * T_INF * np.log(4.6))
