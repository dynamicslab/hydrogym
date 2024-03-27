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

import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor
import ufl

import hydrogym.firedrake as hgym


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
    bcs = flow.collect_bcs()
    q_test = fd.TestFunction(flow.mixed_space)
    Fy, Fx = flow.compute_forces(q=q_test)
    qM = Fy.riesz_representation("L2", bcs=bcs)
    return qM

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
