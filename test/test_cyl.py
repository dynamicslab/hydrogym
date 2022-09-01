import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import time
import pytest


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
        y = flow.collect_observations()
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

    dJdu = fda.compute_gradient(CD, fda.Control(omega))


def test_sensitivity(dt=1e-2, num_steps=10):
    from ufl import inner, dx

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
    dq = fda.compute_gradient(
        J, fda.Control(q0), options={"riesz_representation": "L2"}
    )


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


def solve_omega(torque, I_cm, t):
    if len(I_cm) > 1:
        pytest.fail("Environment with multiple control surfaces not yet supported")
    answer = []
    for I in I_cm:
        answer.append(torque * t / I)
    return answer


def test_no_damp():
    print("")
    print("No Damp")
    time_start = time.time()
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-2
    solver = gym.ts.IPCS(flow, dt=dt, control_method="indirect")
    flow.set_damping(0.0)

    # Apply steady torque for 0.1 seconds... should match analytical solution!
    tf = 0.1  # sec
    torque = 0.05  # Nm
    I_cm = flow.get_inertia()
    analytical_sol = solve_omega(torque, I_cm, tf)

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    print(flow.get_state(), analytical_sol)

    assert np.isclose(flow.get_state(), analytical_sol)

    print("finished @" + str(time.time() - time_start))


def test_fixed_torque():
    print("")
    print("Is Reasonable")
    time_start = time.time()
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-3
    solver = gym.ts.IPCS(flow, dt=dt, control_method="indirect")

    # Apply steady torque of 35.971223 Nm, should converge to ~2 rad/sec with k_damp = 1/TAU
    tf = 1e-2  # sec
    torque = 35.971223  # Nm

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    # damped solved is ~ 3 order of magnitude less than the undamped system, seems high...
    print(flow.get_state())

    assert np.isclose(flow.get_state(), 2.0)

    print("finished @" + str(time.time() - time_start))


def isordered(arr):
    if len(arr) < 2:
        return True
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return False
        return True


# sin function feeding into the controller
def test_convergence_test_varying_torque():
    print("")
    print("Convergence Variable")
    time_start = time.time()
    # larger dt to compensate for longer tf
    dt_list = [1e-2, 5e-3, 2.5e-3, 1e-3]
    dt_baseline = 5e-4

    flow = gym.flow.Cylinder(mesh="coarse")
    solver = gym.ts.IPCS(flow, dt=dt_baseline, control_method="indirect")
    tf = 1e-1  # sec
    torque = 50  # Nm

    # First establish a baseline
    num_steps = int(tf / dt_baseline)
    for iter in range(num_steps):
        # let it run for 3/4 of a period of a sin wave in 0.1 second
        input = np.sin(47.12 * dt_baseline)
        flow = solver.step(iter, control=input)

    baseline_solution = flow.get_state()[0]

    solutions = []
    errors = []

    for dt in dt_list:
        flow = gym.flow.Cylinder(mesh="coarse")
        solver = gym.ts.IPCS(flow, dt=dt, control_method="indirect")

        num_steps = int(tf / dt)
        for iter in range(num_steps):
            input = np.sin(47.2 * dt)
            flow = solver.step(iter, control=input)
        solutions.append(flow.get_state()[0])
        errors.append(np.abs(solutions[-1] - baseline_solution))

    # assert solutions converge to baseline solution
    assert isordered(errors)

    print("finished @" + str(time.time() - time_start))


if __name__ == "__main__":
    test_import()
