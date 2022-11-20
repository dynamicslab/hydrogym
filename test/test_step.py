import firedrake as fd
import firedrake_adjoint as fda
from ufl import sin

import hydrogym.firedrake as hgym


def test_import_coarse():
    flow = hgym.Step(mesh="coarse")
    return flow


def test_import_medium():
    flow = hgym.Step(mesh="medium")
    return flow


def test_import_fine():
    flow = hgym.Step(mesh="fine")
    return flow


def test_steady():
    flow = hgym.Step(Re=100, mesh="coarse")

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_steady_actuation():
    flow = hgym.Step(Re=100, mesh="coarse")
    flow.set_control(1.0)

    solver = hgym.NewtonSolver(flow)
    solver.solve()


def test_integrate():
    flow = hgym.Step(Re=100, mesh="coarse")
    dt = 1e-3

    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS")


def test_integrate_noise():
    flow = hgym.Step(Re=100, mesh="coarse")
    dt = 1e-3

    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS", eta=1.0)


def test_control():
    flow = hgym.Step(Re=100, mesh="coarse")
    dt = 1e-3

    solver = hgym.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow.get_observations()
        flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
    env_config = {
        "flow": hgym.Step,
        "flow_config": {"mesh": "coarse", "Re": 100},
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1 * sin(env.solver.t))


def test_grad():
    flow = hgym.Step(Re=100, mesh="coarse")

    c = fda.AdjFloat(0.0)
    flow.set_control(c)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    (y,) = flow.get_observations()

    dy = fda.compute_gradient(y, fda.Control(c))

    print(dy)
    assert abs(dy) > 0


def test_sensitivity(dt=1e-3, num_steps=10):
    from ufl import dx, inner

    flow = hgym.Step(Re=100, mesh="coarse")

    # Store a copy of the initial condition to distinguish it from the time-varying solution
    q0 = flow.q.copy(deepcopy=True)
    flow.q.assign(
        q0, annotate=True
    )  # Note the annotation flag so that the assignment is tracked

    # Time step forward as usual
    flow = hgym.integrate(flow, t_span=(0, num_steps * dt), dt=dt, method="IPCS_diff")

    # Define a cost functional... here we're just using the energy inner product
    J = 0.5 * fd.assemble(inner(flow.u, flow.u) * dx)

    # Compute the gradient with respect to the initial condition
    #   The option for Riesz representation here specifies that we should end up back in the primal space
    dJ = fda.compute_gradient(
        J, fda.Control(q0), options={"riesz_representation": "L2"}
    )

    assert fd.assemble(inner(dJ, dJ) * dx) > 0


# TODO: Have to add "eta" as a keyword for IPCS_diff
# def test_env_grad():
#     env_config = {"Re": 100, "differentiable": True, "mesh": "coarse"}
#     env = gym.env.StepEnv(env_config)
#     y = env.reset()
#     omega = fd.Constant(1.0)
#     A = fd.Constant(0.1)
#     J = fda.AdjFloat(0.0)
#     for _ in range(10):
#         y, reward, done, info = env.step(A * sin(omega * env.solver.t))
#         J = J - reward
#     dJdm = fda.compute_gradient(J, fda.Control(omega))
#     print(dJdm)
