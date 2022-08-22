import firedrake as fd
import firedrake_adjoint as fda

import hydrogym as gym


def test_import():
    flow = gym.flow.Cylinder(mesh="coarse")
    return flow


def test_import2():
    flow = gym.flow.Cylinder(mesh="medium")
    return flow


def test_import3():
    flow = gym.flow.Cylinder(mesh="fine")
    return flow


def test_steady(tol=1e-3):
    flow = gym.flow.Cylinder(Re=100, mesh="medium")
    flow.solve_steady()

    CL, CD = flow.compute_forces()
    assert abs(CL) < tol
    assert abs(CD - 1.2840) < tol  # Re = 100


def test_rotation(tol=1e-3):
    flow = gym.flow.Cylinder(Re=100, mesh="medium")
    flow.set_control(fd.Constant(0.1))
    flow.solve_steady()

    # Lift/drag on cylinder
    CL, CD = flow.compute_forces()
    assert abs(CL - 0.0594) < tol
    assert abs(CD - 1.2852) < tol  # Re = 100


def test_integrate():
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-2
    gym.ts.integrate(flow, t_span=(0, 10 * dt), dt=dt)


# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K * CL


def test_control():
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-2

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.get_observations()
        flow = solver.step(iter, control=feedback_ctrl(y))


def test_env():
    env_config = {"mesh": "coarse"}
    env = gym.env.CylEnv(env_config)

    u = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(u)
        print(y)
        u = feedback_ctrl(y)


def test_grad():
    flow = gym.flow.Cylinder(mesh="coarse")

    omega = fd.Constant(0.0)
    flow.set_control(omega)

    flow.solve_steady()
    CL, CD = flow.compute_forces()

    fda.compute_gradient(CD, fda.Control(omega))


def test_sensitivity(dt=1e-2, num_steps=10):
    from ufl import dx, inner

    flow = gym.flow.Cylinder(mesh="coarse")

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
    fda.compute_gradient(J, fda.Control(q0), options={"riesz_representation": "L2"})


def test_env_grad():
    env_config = {"differentiable": True, "mesh": "coarse"}
    env = gym.env.CylEnv(env_config)
    y = env.reset()
    K = fd.Constant(0.0)
    J = fda.AdjFloat(0.0)
    for _ in range(10):
        y, reward, done, info = env.step(feedback_ctrl(y, K=K))
        J = J - reward
    dJdm = fda.compute_gradient(J, fda.Control(K))
    print(dJdm)


def test_linearizedNS():
    flow = gym.flow.Cylinder(mesh="coarse")
    qB = flow.solve_steady()
    A, M = flow.linearize(qB, backend="scipy")
    A_adj, M = flow.linearize(qB, adjoint=True, backend="scipy")


if __name__ == "__main__":
    test_import()
