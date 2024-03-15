from ufl import sin
import pytest

import hydrogym.firedrake as hgym

Re_init = [500, 1000, 2000, 4000, 7500]


def test_import_medium():
    hgym.Cavity(Re=500, mesh="medium")


def test_import_fine():
    hgym.Cavity(mesh="fine")


def test_steady():
    flow = hgym.Cavity(Re=50, mesh="medium")
    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_steady_actuation():
    flow = hgym.Cavity(Re=50, mesh="medium")
    flow.set_control(1.0)
    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_integrate():
    flow = hgym.Cavity(Re=50, mesh="medium")
    dt = 1e-4
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt)


def test_control():
    flow = hgym.Cavity(Re=50, mesh="medium")
    dt = 1e-4

    solver = hgym.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow.get_observations()
        flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
    env_config = {
        "flow": hgym.Cavity,
        "flow_config": {"mesh": "medium", "Re": 10},
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1 * sin(env.solver.t))
