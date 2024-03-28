import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
import ufl

from lti_system import control_vec

_alpha_BDF = [1.0, 3.0 / 2.0, 11.0 / 6.0]
_beta_BDF = [
    [1.0],
    [2.0, -1.0 / 2.0],
    [3.0, -3.0 / 2.0, 1.0 / 3.0],
]

MUMPS_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# TODO: Move to lti_system
class LinearBDFSolver:
    def __init__(self, function_space, bilinear_form, dt, bcs=None, f=None, q0=None, order=2, constant_jacobian=True):
        self.function_space = function_space
        self.k = order
        self.h = dt
        self.alpha = _alpha_BDF[order - 1]
        self.beta = _beta_BDF[order - 1]
        self.initialize(function_space, bilinear_form, bcs, f, q0, constant_jacobian)

    def initialize(self, W, A, bcs, f, q0, constant_jacobian):
        if q0 is None:
            q0 = fd.Function(W)
        self.q = q0.copy(deepcopy=True)

        if f is None:
            f = fd.Function(W.sub(0))  # Zero function for RHS forcing

        self.u_prev = [fd.Function(W.sub(0)) for _ in range(self.k)]
        for u in self.u_prev:
            u.assign(self.q.subfunctions[0])

        q_trial = fd.TrialFunction(W)
        q_test = fd.TestFunction(W)
        (u, p) = fd.split(q_trial)
        (v, s) = fd.split(q_test)

        u_BDF = sum(beta * u_n for beta, u_n in zip(self.beta, self.u_prev))

        u_t = (self.alpha * u - u_BDF) / self.h  # BDF estimate of time derivative
        F = ufl.inner(u_t, v) * ufl.dx - (A + ufl.inner(f, v) * ufl.dx)

        a, L = ufl.lhs(F), ufl.rhs(F)
        self.prob = fd.LinearVariationalProblem(a, L, self.q, bcs=bcs, constant_jacobian=constant_jacobian)

        self.solver = fd.LinearVariationalSolver(
            self.prob, solver_parameters=MUMPS_SOLVER_PARAMETERS)

    def step(self):
        self.solver.solve()
        for j in range(self.k - 1):
            idx = self.k - j - 1
            self.u_prev[idx].assign(self.u_prev[idx-1])
        self.u_prev[0].assign(self.q.subfunctions[0])
        return self.q


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

    # Ramp up to k-th order
    for i in range(n_steps):
        q = solver.step()
        flow.q.assign(q)
        CL[i], CD[i] = flow.get_observations()

        if i % 10 == 0:
            hgym.print(f"t={i*h}, CL={CL[i]}, CD={CD[i]}")

    data = np.column_stack((np.arange(n_steps) * h, CL, CD))
    np.save(f"{output_dir}/re{Re}_step_response.npy", data)