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
from ufl import dx, inner

import hydrogym.firedrake as hgym

show_plots = False
output_dir = "output"
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
F = steady_solver.steady_form(qB)  # Nonlinear variational form
J = fd.derivative(F, qB)  # Jacobian with automatic differentiation

flow.linearize_bcs(mixed=True)
flow.set_control([1.0])
bcs = flow.collect_bcs()

# Solve steady, inhomogeneous problem
qC = fd.Function(flow.mixed_space, name="qC")
v, s = fd.TestFunctions(flow.mixed_space)
zero = inner(fd.Constant((0.0, 0.0)), v) * dx
fd.solve(J == zero, qC, bcs=bcs)

if show_plots:
  vort = flow.vorticity(qC.subfunctions[0])
  fig, ax = plt.subplots(1, 1, figsize=(6, 3))
  tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)

with fd.CheckpointFile(f"{output_dir}/lin_fields.h5", "w") as chk:
  chk.save_mesh(flow.mesh)
  chk.save_function(qB)
  chk.save_function(qC)
