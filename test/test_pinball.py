import firedrake as fd
import firedrake_adjoint as fda
import numpy as np
import pyadjoint

import hydrogym.firedrake as hgym


def test_import_coarse():
    flow = hgym.Pinball(mesh="coarse")
    return flow


def test_import_fine():
    flow = hgym.Pinball(mesh="fine")
    return flow


def test_steady(tol=1e-2):
    flow = hgym.Pinball(Re=30, mesh="coarse")
    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL_target = (0.0, 0.520, -0.517)  # Slight asymmetry in mesh
    CD_target = (1.4367, 1.553, 1.554)

    CL, CD = flow.compute_forces()
    for i in range(len(CL)):
        assert abs(CL[i] - CL_target[i]) < tol
        assert abs(CD[i] - CD_target[i]) < tol


def test_steady_rotation(tol=1e-2):
    flow = hgym.Pinball(Re=30, mesh="coarse")
    flow.set_control((0.5, 0.5, 0.5))

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL_target = (0.2718, 0.5035, -0.6276)  # Slight asymmetry in mesh
    CD_target = (1.4027, 1.5166, 1.5696)

    CL, CD = flow.compute_forces()
    print(CL)
    print(CD)
    for i in range(len(CL)):
        assert abs(CL[i] - CL_target[i]) < tol
        assert abs(CD[i] - CD_target[i]) < tol


def test_steady_grad():
    flow = hgym.Pinball(Re=30, mesh="coarse")
    n_cyl = len(flow.CYLINDER)

    # Option 1: List of AdjFloats
    # omega = [fda.AdjFloat(0.5) for _ in range(n_cyl)]
    # control = [fda.Control(omg) for omg in omega]

    # Option 2: List of Constants
    # omega = [fd.Constant(0.5) for i in range(n_cyl)]
    # control = [fda.Control(omg) for omg in omega]

    # # Option 3: Overloaded array with numpy_adjoint
    omega = pyadjoint.create_overloaded_object(np.zeros(n_cyl))
    control = fda.Control(omega)

    flow.set_control(omega)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    J = flow.evaluate_objective()

    # fda.compute_gradient(sum(CD), control)
    dJ = fda.compute_gradient(J, control)
    assert np.all(np.abs(dJ) > 0)


def test_integrate():
    flow = hgym.Pinball(mesh="coarse")
    dt = 1e-2
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt)


def test_integrate_diff():
    flow = hgym.Pinball(mesh="coarse")
    dt = 1e-2
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS_diff")


def test_control():
    flow = hgym.Pinball(mesh="coarse")
    dt = 1e-2

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None:
            K = -0.1 * np.ones((3, 3))  # [Inputs x outputs]
        CL = y[:3]
        return K @ CL

    solver = hgym.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.get_observations()
        flow = solver.step(iter, control=feedback_ctrl(y))


def test_env():
    env_config = {
        "flow": hgym.Pinball,
        "flow_config": {
            "Re": 30,
            "mesh": "coarse",
        },
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None:
            K = -0.1 * np.ones((3, 6))  # [Inputs x outputs]
        return K @ y

    u = np.zeros(env.flow.ACT_DIM)
    for _ in range(10):
        y, reward, done, info = env.step(u)
        u = feedback_ctrl(y)


def test_env_grad():
    # Simple feedback control on lift
    def feedback_ctrl(y, K):
        return K @ y

    env_config = {"Re": 30, "differentiable": True, "mesh": "coarse"}
    env = hgym.env.PinballEnv(env_config)
    y = env.reset()
    n_cyl = env.flow.ACT_DIM
    K = pyadjoint.create_overloaded_object(np.zeros((n_cyl, 2 * n_cyl)))
    J = fda.AdjFloat(0.0)
    for _ in range(10):
        y, reward, done, info = env.step(feedback_ctrl(y, K))
        J = J - reward
    dJ = fda.compute_gradient(J, fda.Control(K))

    assert np.all(np.abs(dJ) > 0)


def test_sensitivity(dt=1e-2, num_steps=10):
    from ufl import dx, inner

    flow = hgym.flow.Pinball(Re=30, mesh="coarse")

    # Store a copy of the initial condition to distinguish it from the time-varying solution
    q0 = flow.q.copy(deepcopy=True)
    flow.q.assign(
        q0, annotate=True
    )  # Note the annotation flag so that the assignment is tracked

    # Time step forward as usual
    flow = hgym.ts.integrate(
        flow, t_span=(0, num_steps * dt), dt=dt, method="IPCS_diff"
    )

    # Define a cost functional... here we're just using the energy inner product
    J = 0.5 * fd.assemble(inner(flow.u, flow.u) * dx)

    # Compute the gradient with respect to the initial condition
    #   The option for Riesz representation here specifies that we should end up back in the primal space
    fda.compute_gradient(J, fda.Control(q0), options={"riesz_representation": "L2"})


# def test_lti():
#     flow = gym.flow.Cylinder()
#     qB = flow.solve_steady()
#     A, M = flow.linearize(qB, backend='scipy')
#     A_adj, M = flow.linearize(qB, adjoint=True, backend='scipy')

if __name__ == "__main__":
    test_import_coarse()
    test_steady_rotation()
    test_steady_grad()
