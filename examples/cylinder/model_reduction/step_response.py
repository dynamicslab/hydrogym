import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
from ufl import inner, dx, lhs, rhs

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

flow = hgym.RotaryCylinder(
    Re=40,
    mesh="medium",
    velocity_order=2,
)

# 1. Solve the steady base flow problem
solver = hgym.NewtonSolver(flow)
qB = solver.solve()

# 2. Derive flow field associated with actuation BC
# See Barbagallo et al. (2009) for details on the "lifting" procedure
F = flow.residual(fd.split(qB))  # Nonlinear variational form
J = fd.derivative(F, qB)  # Jacobian with automatic differentiation

flow.linearize_bcs()
flow.set_control([1.0])
bcs = flow.collect_bcs()

# Solve steady, inhomogeneous problem
qC = fd.Function(flow.mixed_space, name="qC")
v, s = fd.TestFunctions(flow.mixed_space)
zero = inner(fd.Constant((0.0, 0.0)), v) * dx
fd.solve(J == zero, qC, bcs=bcs)


# 3. Step response of the flow
# Second-order BDF time-stepping
A = flow.linearize(qB)
J = A.J

W = A.function_space
bcs = A.bcs
h = 0.01  # Time step

q = flow.q
q.assign(qC)

_alpha_BDF = [1.0, 3.0 / 2.0, 11.0 / 6.0]
_beta_BDF = [
    [1.0],
    [2.0, -1.0 / 2.0],
    [3.0, -3.0 / 2.0, 1.0 / 3.0],
]
k = 2

u_prev = [fd.Function(W.sub(0)) for _ in range(k)]

for u in u_prev:
    u.assign(qC.subfunctions[0])

def order_k_solver(k, W, bcs, u_prev):
    q = flow.q

    k_idx = k - 1
    u_BDF = sum(beta * u_n for beta, u_n in zip(_beta_BDF[k_idx], u_prev[:k]))
    alpha_k = _alpha_BDF[k_idx]

    q_trial = fd.TrialFunction(W)
    q_test = fd.TestFunction(W)
    (u, p) = fd.split(q_trial)
    (v, s) = fd.split(q_test)

    u_t = (alpha_k * u - u_BDF) / h  # BDF estimate of time derivative
    F = inner(u_t, v) * dx - J

    a, L = lhs(F), rhs(F)
    bdf_prob = fd.LinearVariationalProblem(a, L, q, bcs=bcs, constant_jacobian=True)

    MUMPS_SOLVER_PARAMETERS = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    petsc_solver = fd.LinearVariationalSolver(
        bdf_prob, solver_parameters=MUMPS_SOLVER_PARAMETERS)

    def step():
        petsc_solver.solve()
        for j in range(k-1):
            idx = k - j - 1
            u_prev[idx].assign(u_prev[idx-1])
        u_prev[0].assign(q.subfunctions[0])
        return q

    return step

solver = {k: order_k_solver(k, W, bcs, u_prev) for k in range(1, k+1)}


tf = 1000.0
n_steps = int(tf // h)
CL = np.zeros(n_steps)
CD = np.zeros(n_steps)

# Ramp up to k-th order
for i in range(n_steps):
    k_idx = min(i+1, k)
    solver[k_idx]()
    CL[i], CD[i] = flow.get_observations()

    if i % 10 == 0:
        hgym.print(f"t={i*h}, CL={CL[i]}, CD={CD[i]}")

data = np.column_stack((np.arange(n_steps) * h, CL, CD))
np.save(f"{output_dir}/step_response.npy", data)