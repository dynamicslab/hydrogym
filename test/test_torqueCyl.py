import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import time


def solve_omega(torque, I_cm, t):
    return torque * t / I_cm


def test_noDamp():
    print("")
    print("No Damp")
    time_start = time.time()
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-2
    solver = gym.ts.Torque_IPCS(flow, dt=dt)
    solver.set_dampingConst(0.0)

    # Apply steady torque for 0.1 seconds... should match analytical solution!
    tf = 0.1  # sec
    torque = 0.05  # Nm
    I_cm = solver.get_momOfInertia()
    analytical_sol = solve_omega(torque, I_cm, tf)

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    assert np.isclose(solver.get_omega(), analytical_sol)

    print("finished @" + str(time.time() - time_start))


def test_isReasonable():
    print("")
    print("Is Reasonable")
    time_start = time.time()
    flow = gym.flow.Cylinder(mesh="coarse")
    dt = 1e-3
    solver = gym.ts.Torque_IPCS(flow, dt=dt)

    # Apply steady torque of 35.971223 Nm, should converge to ~2 rad/sec with k_damp = 1/TAU
    tf = 1e-2  # sec
    torque = 35.971223  # Nm
    I_cm = solver.get_momOfInertia()
    analytical_sol = solve_omega(torque, I_cm, tf)

    # Run sim
    num_steps = int(tf / dt)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    # damped solved is ~ 3 order of magnitude less than the undamped system, seems high...
    print(solver.get_omega(), analytical_sol)

    assert np.isclose(solver.get_omega(), 2.0)

    print("finished @" + str(time.time() - time_start))


def isordered(arr):
    if len(arr) < 2:
        return True
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return False
        return True


def test_convergenceSteady():
    print("")
    print("Convergence Steady")
    time_start = time.time()
    dt_list = [5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4]
    dt_baseline = 1e-4

    #     # dt_list = [5e-3]
    #     # dt_baseline = 2.5e-3

    flow = gym.flow.Cylinder(mesh="coarse")
    solver = gym.ts.Torque_IPCS(flow, dt=dt_baseline)
    tf = 1e-2  # sec
    torque = 50  # Nm

    # First establish a baseline
    num_steps = int(tf / dt_baseline)
    for iter in range(num_steps):
        flow = solver.step(iter, control=torque)

    baseline_solution = solver.get_omega()

    solutions = []
    errors = []

    for dt in dt_list:
        flow = gym.flow.Cylinder(mesh="coarse")
        solver = gym.ts.Torque_IPCS(flow, dt=dt_baseline)
        num_steps = int(tf / dt)
        for iter in range(num_steps):
            flow = solver.step(iter, control=torque)
        solutions.append(solver.get_omega())
        errors.append(np.abs(solutions[-1] - baseline_solution))

    # assert solutions converge to baseline solution
    assert isordered(errors)

    print("finished @" + str(time.time() - time_start))


# sin function feeding into the controller
def test_convergenceVariable():
    print("")
    print("Convergence Variable")
    time_start = time.time()
    # larger dt to compensate for longer tf
    dt_list = [1e-2, 5e-3, 2.5e-3, 1e-3]
    dt_baseline = 5e-4

    # dt_list = [1e-2]
    # dt_baseline = 5e-3

    flow = gym.flow.Cylinder(mesh="coarse")
    solver = gym.ts.Torque_IPCS(flow, dt=dt_baseline)
    tf = 1e-1  # sec
    torque = 50  # Nm

    # First establish a baseline
    num_steps = int(tf / dt_baseline)
    for iter in range(num_steps):
        # let it run for 3/4 of a period of a sin wave in 0.1 second
        input = np.sin(47.12 * dt_baseline)
        flow = solver.step(iter, control=input)

    baseline_solution = solver.get_omega()

    solutions = []
    errors = []

    for dt in dt_list:
        flow = gym.flow.Cylinder(mesh="coarse")
        solver = gym.ts.Torque_IPCS(flow, dt=dt)

        num_steps = int(tf / dt)
        for iter in range(num_steps):
            input = np.sin(47.2 * dt)
            flow = solver.step(iter, control=input)
        solutions.append(solver.get_omega())
        errors.append(np.abs(solutions[-1] - baseline_solution))

    # assert solutions converge to baseline solution
    assert isordered(errors)

    print("finished @" + str(time.time() - time_start))


def test_shearForce0():
    flow = gym.flow.Cylinder(Re=100, mesh="coarse")
    flow.set_control(fd.Constant(0.0))
    flow.solve_steady()
    shear_force = flow.shear_force()

    assert np.isclose(shear_force, 0.0, rtol=1e-3, atol=1e-3)


def test_shearForcePos():
    flow = gym.flow.Cylinder(Re=100, mesh="coarse")
    flow.set_control(fd.Constant(0.1))
    flow.solve_steady()
    shear_force = flow.shear_force()

    assert shear_force < 0


def test_shearForceNeg():
    flow = gym.flow.Cylinder(Re=100, mesh="coarse")
    flow.set_control(fd.Constant(-0.1))
    flow.solve_steady()
    shear_force = flow.shear_force()

    assert shear_force > 0


# TODO: verify that the shear force is correct in magnitude... method to do this is unclear


# test that it doesn't break in long simulations
# Not necessary in my opinion
# def test_endurance():
#     flow = gym.flow.Cylinder(mesh='coarse')
#     dt = 1e-2
#     solver = gym.ts.Torque_IPCS(flow, dt=dt)

#     # Apply steady toque for one second... should be slightly less than  analytical solution!
#     tf = 50 #sec
#     torque = 20 #Nm

#     # Run sim
#     num_steps = int(tf / dt)
#     for iter in range(num_steps):
#         flow = solver.step(iter, control = torque)
#         print(solver.get_omega())

#     print(solver.get_omega())

if __name__ == "__main__":
    test_import()
