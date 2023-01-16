import firedrake as fd
import firedrake_adjoint as fda
from ufl import sin

import hydrogym.firedrake as hgym


def test_import_coarse():
    hgym.Cavity(mesh="coarse")


def test_import_medium():
    hgym.Cavity(mesh="medium")


def test_import_fine():
    hgym.Cavity(mesh="fine")


def test_steady():
    flow = hgym.Cavity(Re=50, mesh="coarse")

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_steady_actuation():
    flow = hgym.Cavity(Re=50, mesh="coarse")
    flow.set_control(1.0)

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_integrate():
    flow = hgym.Cavity(Re=50, mesh="coarse")
    dt = 1e-4

    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt)


def test_control():
    flow = hgym.Cavity(Re=50, mesh="coarse")
    dt = 1e-4

    solver = hgym.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow.get_observations()
        flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
    env_config = {
        "flow": hgym.Cavity,
        "flow_config": {"mesh": "coarse", "Re": 10},
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1 * sin(env.solver.t))


def test_grad():
    flow = hgym.Cavity(Re=50, mesh="coarse")

    c = fda.AdjFloat(0.0)
    flow.set_control(c)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    (y,) = flow.get_observations()

    dy = fda.compute_gradient(y, fda.Control(c))

    print(dy)
    assert abs(dy) > 0


if __name__ == "__main__":
    test_import_medium()
