import numpy as np
import hydrogym.firedrake as hgym


def test_import_fine():
    hgym.Pinball(mesh="fine")


def test_steady(tol=1e-2):
    flow = hgym.Pinball(Re=30, mesh="fine")
    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL_target = (0.0, 0.521, -0.521)  # Slight asymmetry in mesh
    CD_target = (1.451, 1.566, 1.566)

    CL, CD = flow.compute_forces()
    for i in range(len(CL)):
        assert abs(CL[i] - CL_target[i]) < tol
        assert abs(CD[i] - CD_target[i]) < tol


def test_steady_rotation(tol=1e-2):
    flow = hgym.Pinball(Re=30, mesh="fine")
    flow.set_control((0.5, 0.5, 0.5))

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL_target = (-0.2477, 0.356, -0.6274)
    CD_target = (1.4476, 1.6887, 1.4488)

    CL, CD = flow.compute_forces()
    for i in range(len(CL)):
        assert abs(CL[i] - CL_target[i]) < tol
        assert abs(CD[i] - CD_target[i]) < tol


def test_integrate():
    flow = hgym.Pinball(mesh="fine")
    dt = 1e-2
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt)


def test_control():
    flow = hgym.Pinball(mesh="fine")
    dt = 1e-2

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None:
            K = -0.1 * np.ones((3, 3))  # [Inputs x outputs]
        CL = y[:3]
        return K @ CL

    solver = hgym.SemiImplicitBDF(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.get_observations()
        flow = solver.step(iter, control=feedback_ctrl(y))


def test_env():
    env_config = {
        "flow": hgym.Pinball,
        "flow_config": {
            "Re": 30,
            "mesh": "fine",
        },
        "solver": hgym.SemiImplicitBDF,
    }
    env = hgym.FlowEnv(env_config)

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None:
            K = -0.1 * np.ones((3, 6))  # [Inputs x outputs]
        return K @ y

    u = np.zeros(len(env.flow.CYLINDER))
    for _ in range(10):
        y, reward, done, info = env.step(u)
        u = feedback_ctrl(y)
