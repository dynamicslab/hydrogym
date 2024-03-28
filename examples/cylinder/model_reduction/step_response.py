import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
import ufl

from lti_system import control_vec, LinearBDFSolver


if __name__ == "__main__":
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Re = 100

    flow = hgym.RotaryCylinder(
        Re=Re,
        mesh="medium",
        velocity_order=2,
    )

    # 1. Solve the steady base flow problem
    solver = hgym.NewtonSolver(flow)
    qB = solver.solve()

    # 2. Derive flow field associated with actuation BC
    # See Barbagallo et al. (2009) for details on the "lifting" procedure
    qC = control_vec(flow)


    # 3. Step response of the flow
    # Second-order BDF time-stepping
    A = flow.linearize(qB)
    J = A.J

    W = A.function_space
    bcs = A.bcs
    h = 0.01  # Time step

    solver = LinearBDFSolver(W, J, h, bcs, qC, order=2)

    tf = 100.0
    n_steps = int(tf // h)
    CL = np.zeros(n_steps)
    CD = np.zeros(n_steps)

    for i in range(n_steps):
        q = solver.step()
        flow.q.assign(q)
        CL[i], CD[i] = flow.get_observations()

        if i % 10 == 0:
            hgym.print(f"t={i*h}, CL={CL[i]}, CD={CD[i]}")

    data = np.column_stack((np.arange(n_steps) * h, CL, CD))
    np.save(f"{output_dir}/re{Re}_step_response.npy", data)
