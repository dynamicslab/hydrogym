import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import pyadjoint
from ufl import sin


def test_import():
    flow = gym.flow.Cavity(mesh="medium")
    return flow


def test_import2():
    flow = gym.flow.Cavity(mesh="fine")
    return flow


def test_steady(tol=1e-3):
    flow = gym.flow.Cavity(Re=500, mesh="medium")
    flow.solve_steady()

    y = flow.collect_observations()
    assert abs(y - 2.2122) < tol  # Re = 500


def test_actuation():
    flow = gym.flow.Cavity(Re=500, mesh="medium")
    flow.set_control(1.0)
    flow.solve_steady()


def test_step():
    flow = gym.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow = solver.step(iter)


def test_integrate():
    flow = gym.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    gym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS")


def test_control():
    flow = gym.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.collect_observations()
        flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
    env_config = {"mesh": "medium"}
    env = gym.env.CavityEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1 * sin(env.solver.t))


def test_grad():
    flow = gym.flow.Cavity(Re=500, mesh="medium")

    c = fd.Constant(0.0)
    flow.set_control(c)

    flow.solve_steady()
    y = flow.collect_observations()

    dJdu = fda.compute_gradient(y, fda.Control(c))


def test_sensitivity(dt=1e-2, num_steps=10):
    from ufl import inner, dx

    flow = gym.flow.Cavity(Re=5000, mesh="medium")

    # Store a copy of the initial condition to distinguish it from the time-varying solution
    q0 = flow.q.copy(deepcopy=True)
    flow.q.assign(
        q0, annotate=True
    )  # Note the annotation flag so that the assignment is tracked

    # Time step forward as usual
    flow = gym.ts.integrate(flow, t_span=(0, num_steps * dt), dt=dt, method="IPCS_diff")

    # Define a cost functional... here we're just using the energy inner product
    J = 0.5 * fd.assemble(inner(flow.u, flow.u) * dx)

    # Compute the gradient with respect to the initial condition
    #   The option for Riesz representation here specifies that we should end up back in the primal space
    dq = fda.compute_gradient(
        J, fda.Control(q0), options={"riesz_representation": "L2"}
    )


def test_env_grad():
    env_config = {"differentiable": True, "mesh": "medium"}
    env = gym.env.CavityEnv(env_config)
    y = env.reset()
    omega = fd.Constant(1.0)
    A = fd.Constant(0.1)
    J = fda.AdjFloat(0.0)
    for _ in range(10):
        y, reward, done, info = env.step(A * sin(omega * env.solver.t))
        J = J - reward
    dJdm = fda.compute_gradient(J, fda.Control(omega))
    print(dJdm)


if __name__ == "__main__":
    test_import()
