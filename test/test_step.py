from ufl import sin

import hydrogym.firedrake as hgym


def test_import_medium():
    hgym.Step(mesh="medium")


def test_import_fine():
    hgym.Step(mesh="fine")


def test_steady():
    flow = hgym.Step(Re=100, mesh="medium")

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_steady_actuation():
    flow = hgym.Step(Re=100, mesh="medium")
    flow.set_control(1.0)

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_integrate():
    flow = hgym.Step(Re=100, mesh="medium")
    dt = 1e-3

    hgym.integrate(
        flow,
        t_span=(0, 10 * dt),
        dt=dt,
    )


def test_integrate_noise():
    flow = hgym.Step(Re=100, mesh="medium")
    dt = 1e-3

    hgym.integrate(
        flow,
        t_span=(0, 10 * dt),
        dt=dt,
        eta=1.0
    )


def test_control():
    flow = hgym.Step(Re=100, mesh="medium")
    dt = 1e-3

    solver = hgym.SemiImplicitBDF(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow.get_observations()
        flow = solver.step(iter, control=0.1 * sin(iter * solver.dt))


def test_env():
    env_config = {
        "flow": hgym.Step,
        "flow_config": {
            "mesh": "medium",
            "Re": 100
        },
        "solver": hgym.SemiImplicitBDF,
    }
    env = hgym.FlowEnv(env_config)

    for i in range(10):
        y, reward, done, info = env.step(0.1 * sin(i * env.solver.dt))
