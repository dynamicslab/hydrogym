import numpy as np
import pytest

pytest.importorskip("firedrake")

import firedrake as fd  # noqa: E402

import hydrogym.firedrake as hgym  # noqa: E402
from hydrogym.firedrake.utils.pd import PDController  # noqa: E402


def test_import_medium():
    hgym.Cylinder(mesh="medium")


def test_import_fine():
    hgym.Cylinder(mesh="fine")


def test_steady(tol=1e-3):
    # Newton solve from zero initial guess (no HF checkpoint); reference values CL = 0.000039, CD = 1.284043.
    flow = hgym.Cylinder(Re=100, mesh="medium", use_HF_data_manager=False)
    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL, CD = flow.compute_forces()
    assert abs(CL) < tol
    assert abs(CD - 1.2840) < tol  # Re = 100


def test_probes_in_given_order(tol=1e-10):
    # Linear fields are represented exactly, so each probe must return its own coordinates, in the order given
    probes = [(6.0, 1.5), (1.0, -2.0), (3.0, 0.5), (9.0, -1.0), (1.5, 1.0), (4.0, -0.5)]
    flow = hgym.Cylinder(mesh="medium", use_HF_data_manager=False)
    x, y = fd.SpatialCoordinate(flow.mesh)
    flow.u.interpolate(fd.as_vector((x, y)))
    flow.p.interpolate(x + 2 * y)
    xy = np.asarray(probes)
    assert np.allclose(flow.velocity_probe(probes), np.concatenate([xy[:, 0], xy[:, 1]]), atol=tol)
    assert np.allclose(flow.pressure_probe(probes), xy[:, 0] + 2 * xy[:, 1], atol=tol)


def test_jet_zero_net_mass_flux(tol=1e-8):
    flow = hgym.Cylinder(mesh="medium")

    # Volume flux from the cylinder wall into the fluid for unit actuation
    flux = fd.dot(flow.cyl_velocity_field, -flow.n)
    upper_flux = fd.assemble(fd.conditional(flow.y > 0, flux, 0.0) * fd.ds(flow.CYLINDER))
    lower_flux = fd.assemble(fd.conditional(flow.y < 0, flux, 0.0) * fd.ds(flow.CYLINDER))

    assert upper_flux > 0  # Blowing through the upper jet
    assert abs(upper_flux + lower_flux) < tol * abs(upper_flux)


def test_jet_legacy_same_sign(tol=1e-8):
    """``legacy_jets`` (HydroGym <= 1.0): both jets blow with the same flow rate (net flow, not ZNMF)."""
    flow = hgym.Cylinder(mesh="medium", legacy_jets=True)

    flux = fd.dot(flow.cyl_velocity_field, -flow.n)
    upper_flux = fd.assemble(fd.conditional(flow.y > 0, flux, 0.0) * fd.ds(flow.CYLINDER))
    lower_flux = fd.assemble(fd.conditional(flow.y < 0, flux, 0.0) * fd.ds(flow.CYLINDER))

    assert upper_flux > 0
    assert abs(upper_flux - lower_flux) < tol * abs(upper_flux)


@pytest.mark.parametrize(
    "legacy, CL_ref, CD_ref",
    [
        (False, -0.195378, 1.453947),
        # legacy_startup_order: the off-by-one BDF start-up of HydroGym <= 1.0
        (True, -0.195866, 1.455184),
    ],
)
def test_steady_rotation(legacy, CL_ref, CD_ref, tol=1e-3):
    Tf = 4.0
    dt = 0.1

    # Starts from zero; reference values measured with this procedure.
    flow = hgym.RotaryCylinder(Re=100, mesh="medium", use_HF_data_manager=False)
    flow.set_control(0.1)

    solver = hgym.SemiImplicitBDF(flow, dt=dt, legacy_startup_order=legacy)

    for iter in range(int(Tf / dt)):
        solver.step(iter)

    # Lift/drag on cylinder
    CL, CD = flow.compute_forces()
    assert abs(CL - CL_ref) < tol
    assert abs(CD - CD_ref) < tol  # Re = 100


@pytest.mark.parametrize("k", [1, 3])
def test_integrate(k):
    flow = hgym.Cylinder(mesh="medium")
    dt = 1e-2
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, order=k)


@pytest.mark.parametrize(
    "k, legacy, expected",
    [
        (1, False, [1, 1]),
        (2, False, [1, 2, 2]),
        (3, False, [1, 2, 3, 3]),
        # legacy_startup_order: the off-by-one sequence of HydroGym <= 1.0
        (1, True, [1, 1]),
        (2, True, [1, 1, 2]),
        (3, True, [2, 1, 2, 3]),
    ],
)
def test_bdf_startup_order(k, legacy, expected):
    flow = hgym.Cylinder(mesh="medium")
    solver = hgym.SemiImplicitBDF(flow, dt=1e-2, order=k, legacy_startup_order=legacy)

    # Record the order of the scheme used for each step
    orders = []
    for order, petsc_solver in enumerate([*solver.startup_solvers, solver.petsc_solver], start=1):

        def solve(order=order, solve=petsc_solver.solve):
            orders.append(order)
            return solve()

        petsc_solver.solve = solve

    for iter in range(k + 1):
        solver.step(iter)

    assert orders == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_env_substeps_integrate_like_single_steps(legacy):
    """2 actions x 3 substeps integrate exactly like 6 single-substep actions (same BDF orders, same state).

    With ``legacy_substep_iteration`` (HydroGym <= 1.0) the start-up solvers run again at the second action.
    """

    def make(num_substeps):
        return hgym.FlowEnv(
            {
                "flow": hgym.Cylinder,
                "flow_config": {"mesh": "medium"},
                "solver": hgym.SemiImplicitBDF,
                "solver_config": {"dt": 1e-2, "legacy_startup_order": legacy},
                "actuation_config": {"num_substeps": num_substeps, "legacy_substep_iteration": legacy},
            }
        )

    single, multi = make(1), make(3)
    single.reset()
    multi.reset()
    for _ in range(6):
        single.step(0.1)
    for _ in range(2):
        multi.step(0.1)
    assert single.iter == multi.iter == 6
    u1, u3 = single.flow.u.dat.data_ro, multi.flow.u.dat.data_ro  # same mesh file, two Mesh objects
    diff = np.abs(u1 - u3).max() / np.abs(u1).max()
    if legacy:
        assert diff > 1e-8
    else:
        assert diff < 1e-12


# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K * CL


def test_control():
    k = 2.0
    theta = 4.0
    tf = 10.0
    dt = 1e-2

    # Initialize the flow
    flow = hgym.RotaryCylinder(Re=100, mesh="medium", velocity_order=1)

    # Construct the PDController
    pd_controller = PDController(
        k * np.cos(theta),
        k * np.sin(theta),
        1e-2,
        int(10.0 // 1e-2) + 2,
        filter_type="bilinear",
        N=2,
    )

    def controller(t, obs):
        if t < tf / 2:
            return 0.0
        return pd_controller(t, obs)

    hgym.integrate(flow, t_span=(0, tf), dt=dt, controller=controller)


def test_env():
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "medium",
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
        },
    }
    env = hgym.FlowEnv(env_config)

    u = 0.0
    for _ in range(10):
        y, reward, terminated, truncated, info = env.step(u)
        u = feedback_ctrl(y)

    env.reset()


def test_linearize():
    # use_HF_data_manager=False: same checkpoint-ambiguity issue as
    # test_steady above -- a Newton steady solve needs a near-equilibrium
    # initial guess, not an arbitrary transient snapshot auto-inferred from
    # the Hub checkpoint (see test_steady's comment).
    flow = hgym.Cylinder(mesh="medium", use_HF_data_manager=False)

    solver = hgym.NewtonSolver(flow)
    qB = solver.solve()

    A, M = hgym.modeling.linearize(flow, qB, backend="scipy")
    A_adj, M = hgym.modeling.linearize(flow, qB, adjoint=True, backend="scipy")


def test_act_implicit_no_damp():
    # use_HF_data_manager=False: see test_steady's comment -- same Newton
    # steady solve, same checkpoint-ambiguity issue with the auto-inferred
    # Cylinder_2D_Re100_medium_FD environment.
    flow = hgym.Cylinder(mesh="medium", actuator_integration="implicit", use_HF_data_manager=False)
    # dt = 1e-2
    solver = hgym.NewtonSolver(flow)

    # Experimental: modify the actuator attributes after the solver is created
    flow.actuators[0].k = 0
    solver.solve()


def test_act_implicit_fixed_torque():
    dt = 1e-4

    # Define the flow
    flow = hgym.Cylinder(mesh="medium", actuator_integration="implicit")

    # Set up the solver
    solver = hgym.SemiImplicitBDF(flow, dt=dt)

    # Obtain a torque value for which the system converges to a steady state angular velocity
    tf = 0.1 * flow.TAU
    omega = 1.0

    # Torque to reach steady-state value of `omega`
    torque = omega / flow.TAU

    # Run sim
    num_steps = 10 * int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)
