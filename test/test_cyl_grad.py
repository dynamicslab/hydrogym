def isordered(arr):
    if len(arr) < 2:
        return True
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return False
        return True


"""
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
"""

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
