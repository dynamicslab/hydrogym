import time

import firedrake as fd
import firedrake_adjoint as fda
import numpy as np

import hydrogym.firedrake as hgym


def test_import_coarse():
    hgym.Cylinder(mesh="coarse")


def test_import_medium():
    hgym.Cylinder(mesh="medium")


def test_import_fine():
    hgym.Cylinder(mesh="fine")


def test_steady(tol=1e-3):
    flow = hgym.Cylinder(Re=100, mesh="medium")
    solver = hgym.NewtonSolver(flow)
    solver.solve()

    CL, CD = flow.compute_forces()
    assert abs(CL) < tol
    assert abs(CD - 1.2840) < tol  # Re = 100


def test_steady_rotation(tol=1e-3):
    flow = hgym.Cylinder(Re=100, mesh="medium")
    flow.set_control(0.1)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    # Lift/drag on cylinder
    CL, CD = flow.compute_forces()
    assert abs(CL - 0.0594) < tol
    assert abs(CD - 1.2852) < tol  # Re = 100


def test_steady_grad():
    flow = hgym.Cylinder(Re=100, mesh="coarse")

    # First test with AdjFloat
    omega = fda.AdjFloat(0.1)

    flow.set_control(omega)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    J = flow.evaluate_objective()
    dJ = fda.compute_gradient(J, fda.Control(omega))

    assert abs(dJ) > 0


def test_integrate():
    flow = hgym.Cylinder(mesh="coarse")
    dt = 1e-2
    hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt)


# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K * CL


def test_control():
    flow = hgym.Cylinder(mesh="coarse")
    dt = 1e-2

    solver = hgym.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.get_observations()
        hgym.print(y)
        flow = solver.step(iter, control=feedback_ctrl(y))


def test_env():
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "coarse",
        },
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    u = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(u)
        print(y)
        u = feedback_ctrl(y)


def test_linearize():
    flow = hgym.Cylinder(mesh="coarse")

    solver = hgym.NewtonSolver(flow)
    qB = solver.solve()

    A, M = hgym.modeling.linearize(flow, qB, backend="scipy")
    A_adj, M = hgym.modeling.linearize(flow, qB, adjoint=True, backend="scipy")


def test_no_damp():
    print("")
    print("No Damp")
    time_start = time.time()
    flow = hgym.Cylinder(mesh="coarse", control_method="indirect")
    dt = 1e-2
    solver = hgym.IPCS(flow, dt=dt)

    # Since this feature is still experimental, modify actuator attributes *after*
    flow.actuators[0].implicit = True
    flow.actuators[0].k = 0

    # Apply steady torque for 0.1 seconds... should match analytical solution!
    tf = 0.1  # sec
    torque = 0.05  # Nm
    analytical_sol = [torque * tf / flow.I_CM]

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    print(flow.control_state, analytical_sol)

    final_torque = fd.Constant(flow.control_state[0]).values()
    assert np.isclose(final_torque, analytical_sol)

    print("finished @" + str(time.time() - time_start))


def test_fixed_torque():
    print("")
    print("Fixed Torque Convergence")
    time_start = time.time()
    flow = hgym.Cylinder(mesh="coarse", control_method="indirect")
    dt = 1e-3
    solver = hgym.IPCS(flow, dt=dt)
    flow.actuators[0].implicit = True

    # Obtain a torque value for which the system converges to a steady state angular velocity
    tf = 0.1 * flow.TAU
    omega = 1.0
    torque = omega / flow.TAU  # Torque to reach steady-state value of `omega`

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)
        print(flow.control_state[0].values())

    final_torque = fd.Constant(flow.control_state[0]).values()
    assert np.isclose(final_torque, omega, atol=1e-3)

    print("finished @" + str(time.time() - time_start))


def isordered(arr):
    if len(arr) < 2:
        return True
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return False
        return True


# sin function feeding into the controller
# TODO: This is too big for a unit test - should go in examples or somewhere like that
# def test_convergence_test_varying_torque():
#     print("")
#     print("Convergence Test with Varying Torque Commands")
#     time_start = time.time()
#     # larger dt to compensate for longer tf
#     dt_list = [1e-2, 5e-3, 2.5e-3, 1e-3]
#     dt_baseline = 5e-4

#     flow = gym.flow.Cylinder(mesh="coarse", control_method="indirect")
#     solver = gym.ts.IPCS(flow, dt=dt_baseline)
#     tf = 1e-1  # sec
#     # torque = 50  # Nm

#     # First establish a baseline
#     num_steps = int(tf / dt_baseline)
#     for iter in range(num_steps):
#         # let it run for 3/4 of a period of a sin wave in 0.1 second
#         input = np.sin(47.12 * dt_baseline)
#         flow = solver.step(iter, control=input)

#     baseline_solution = flow.get_ctrl_state()[0]

#     solutions = []
#     errors = []

#     for dt in dt_list:
#         flow = gym.flow.Cylinder(mesh="coarse", control_method="indirect")
#         solver = gym.ts.IPCS(flow, dt=dt)

#         num_steps = int(tf / dt)
#         for iter in range(num_steps):
#             input = np.sin(47.2 * dt)
#             flow = solver.step(iter, control=input)
#         solutions.append(flow.get_ctrl_state()[0])
#         errors.append(np.abs(solutions[-1] - baseline_solution))

#     # assert solutions converge to baseline solution
#     assert isordered(errors)

#     print("finished @" + str(time.time() - time_start))


# Shear force test cases (shear force not fully implemented yet)

# def test_shearForce0():
#     flow = gym.flow.Cylinder(Re=100, mesh="coarse")
#     flow.set_control(fd.Constant(0.0))
#     flow.solve_steady()
#     shear_force = flow.shear_force()

#     assert np.isclose(shear_force, 0.0, rtol=1e-3, atol=1e-3)


# def test_shearForcePos():
#     flow = gym.flow.Cylinder(Re=100, mesh="coarse")
#     flow.set_control(fd.Constant(0.1))
#     flow.solve_steady()
#     shear_force = flow.shear_force()

#     assert shear_force < 0


# def test_shearForceNeg():
#     flow = gym.flow.Cylinder(Re=100, mesh="coarse")
#     flow.set_control(fd.Constant(-0.1))
#     flow.solve_steady()
#     shear_force = flow.shear_force()

#     assert shear_force > 0

if __name__ == "__main__":
    # test_no_damp()
    # test_steady_grad()
    # test_control()
    test_fixed_torque()
