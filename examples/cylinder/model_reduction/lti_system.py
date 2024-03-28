"""Illustrate constructing an LTI system from a flow configuration.

This is a work in progress - so far it's just the base flow and the control
vector. Ultimately it will need LinearOperator functionality to have the form

````
x' = A*x + B*u
y  = C*x + D*u
```

where `x` is a firedrake.Function and `u` and `y` are numpy arrays.
"""
import os

import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor
import ufl

import hydrogym.firedrake as hgym

MUMPS_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


_alpha_BDF = [1.0, 3.0 / 2.0, 11.0 / 6.0]
_beta_BDF = [
    [1.0],
    [2.0, -1.0 / 2.0],
    [3.0, -3.0 / 2.0, 1.0 / 3.0],
]


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


def control_vec(flow):
    """Derive flow field associated with actuation BC

    See Barbagallo et al. (2009) for details on the "lifting" procedure
    """
    qB = flow.q
    F = flow.residual(fd.split(qB))  # Nonlinear variational form
    J = fd.derivative(F, qB)  # Jacobian with automatic differentiation

    flow.linearize_bcs()
    flow.set_control([1.0])
    bcs = flow.collect_bcs()

    # Solve steady, inhomogeneous problem
    qC = fd.Function(flow.mixed_space, name="qC")
    v, s = fd.TestFunctions(flow.mixed_space)
    zero = ufl.inner(fd.Constant((0.0, 0.0)), v) * ufl.dx
    fd.solve(J == zero, qC, bcs=bcs)
    return qC


def measurement_matrix(flow):
    """Derive measurement matrix for flow field.

    This is a field qM such that the inner product of qM with the flow field
    produces the same result as computing the observation (lift coefficient)
    
    This implementation is specific to the cylinder, but could be generalized
    """
    flow.linearize_bcs()
    q_test = fd.TestFunction(flow.mixed_space)
    Fy, Fx = flow.compute_forces(q=q_test)
    qM = Fy.riesz_representation("L2")
    return qM


def make_transfer_function(lin_flow, qC, qM):
  """Create a transfer function H(s) = C(sI - A)^{-1} B + D"""
  # Feedthrough term
  D = fd.assemble(
    sum(ufl.inner(uM, uC) for (uM, uC) in zip(qM, qC)) * ufl.dx
  )

  # Solve (sM - A) q = B for q = (sI - A)^{-1} B
  A = lin_flow.J
  M = lin_flow.M
  q = fd.Function(lin_flow.function_space)
  _s = fd.Constant(0.0)

  problem = fd.LinearVariationalProblem(
    _s * M - A,
    ufl.action(M, qC),
    q,
    bcs=lin_flow.bcs,
  )

  solver = fd.LinearVariationalSolver(
    problem, solver_parameters=MUMPS_SOLVER_PARAMETERS
  )

  def transfer_function(s):
    _s.assign(s)
    solver.solve()
    H_ex_D = fd.assemble(
        sum(ufl.inner(uM, q) for (uM, q) in zip(qM, q)) * ufl.dx
    )
    return H_ex_D + D

  return transfer_function


if __name__ == "__main__":
  show_plots = False
  output_dir = "lti_output"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  flow = hgym.RotaryCylinder(Re=100, mesh="medium")

  # 1. Compute base flow
  steady_solver = hgym.NewtonSolver(flow)
  qB = steady_solver.solve()
  qB.rename("qB")

  # Check lift/drag
  print(flow.compute_forces(qB))

  if show_plots:
    vort = flow.vorticity(qB.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)

  # 2. Derive flow field associated with actuation BC
  # See Barbagallo et al. (2009) for details on the "lifting" procedure
  qC = control_vec(flow)

  if show_plots:
    vort = flow.vorticity(qC.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)

  # 3. Derive measurement matrix for flow field.
  # This is a field qM such that the inner product of qM with the flow field
  # produces the same result as computing the observation (lift coefficient)
  qM = measurement_matrix(flow)

  # 4. Sanity check on the measurement matrix: compute the feedthrough term D
  # First by directly computing the forces from the flow
  flow.q.assign(qC)
  CL, CD = flow.compute_forces()
  
  # Then by computing the inner product of the measurement matrix with the flow field
  D = fd.assemble(
      sum(ufl.inner(uM, uC) for (uM, uC) in zip(qM, qC)) * ufl.dx
  )

  print(f"{CL=}, {D=}")

  with fd.CheckpointFile(f"{output_dir}/lin_fields.h5", "w") as chk:
    chk.save_mesh(flow.mesh)
    qB.rename("qB")
    chk.save_function(qB)
    qC.rename("qC")
    chk.save_function(qC)
    qM.rename("qM")
    chk.save_function(qM)


  # # 5. Create a transfer function H(s) = C(sI - A)^{-1} B + D
  # lin_flow = flow.linearize(qB)
  # transfer_function = make_transfer_function(lin_flow, qC, qM)

  # omega = np.linspace(0.0001, 4.0, 1000)
  # H = np.zeros(omega.shape, dtype=complex)
  # for i, s in enumerate(1j * omega):
  #     H[i] = transfer_function(s)
  #     print(f"{s} -> {abs(H[i])}")

  # np.savez(f"{output_dir}/transfer_function", omega=omega, H=H)