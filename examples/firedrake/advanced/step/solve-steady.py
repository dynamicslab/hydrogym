#!/usr/bin/env python3
"""
Backward-Facing Step - Steady State Solver

Solve for steady-state flow over backward-facing step at Re=600.
Uses Newton iteration with Reynolds ramping for robust convergence.

Usage:
    python solve-steady.py

Physical setup:
    - Backward-facing step with sudden expansion
    - Reynolds number Re = 600
    - Separation and reattachment zones
    - Outputs: Steady velocity, pressure, vorticity fields

Note: The steady state at Re=600 may be unstable, exhibiting vortex shedding
      when perturbed. This solver finds the equilibrium solution useful for
      stability studies and as a base flow.
"""
import os
import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
mesh_resolution = "fine"
Re = 600  # Reynolds number
checkpoint_prefix = f"{output_dir}/{Re}_steady"

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
# Gradually increase Re from 100 to target value in steps of 100
Re_init = np.arange(100, Re + 100, 100, dtype=float)

# Create flow configuration with P1-P1 elements
flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=1,  # P1 velocity elements
    use_HF_data_manager=False,
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

# Set up Newton solver for nonlinear steady-state problem
# GLS stabilization needed for P1-P1 elements
solver = hgym.NewtonSolver(
    flow,
    stabilization="gls",  # Galerkin Least Squares for P1-P1 stability
    solver_parameters=solver_parameters,
)

# Reynolds ramping loop - gradually increase Re to target value
for i, Re in enumerate(Re_init):
  flow.Re.assign(Re)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  qB = solver.solve()

# Save steady state checkpoint for restart or post-processing
flow.save_checkpoint(f"{checkpoint_prefix}.h5")

# Save visualization with velocity, pressure, and vorticity
vort = flow.vorticity()
try:
  # Try newer Firedrake API
  pvd = fd.VTKFile(f"{checkpoint_prefix}.pvd")
  pvd.write(flow.u, flow.p, vort)
except AttributeError:
  # Fall back to older API
  pvd = fd.File(f"{checkpoint_prefix}.pvd")
  pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {checkpoint_prefix}.h5")
