#!/usr/bin/env python3
"""
Cavity Flow - Steady State Solver

Solve for steady-state flow in open cavity at Re=7500.
Uses Newton iteration with Reynolds ramping for robust convergence at high Re.

Usage:
    python solve-steady.py

Physical setup:
    - Open square cavity (1×1) with moving top wall
    - Inlet velocity: U = 1.0
    - Reynolds number Re = 7500 (turbulent regime)
    - Outputs: Steady velocity, pressure, vorticity fields

Note: At Re=7500, the flow is turbulent, but this solver finds the unstable
      steady state solution, which is useful for stability analysis and as
      a base flow for turbulence studies.
"""

import os
import firedrake as fd

import hydrogym.firedrake as hgym

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
mesh_resolution = "fine"
Re = 7500  # High Reynolds number - requires careful ramping

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
# At high Re, direct solve often fails - start low and gradually increase
Re_init = [500, 1000, 2000, 4000, Re]

# Create flow configuration with P1-P1 elements
flow = hgym.Cavity(Re=Re_init[0], mesh=mesh_resolution, velocity_order=1, use_HF_data_manager=False)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

# Set up Newton solver for nonlinear steady-state problem
# Note: P1-P1 elements require stabilization in general, but GLS is optional here
solver = hgym.NewtonSolver(
    flow,
    stabilization="none",
    solver_parameters=solver_parameters,
)

# Reynolds ramping loop - gradually increase Re to target value
for i, Re in enumerate(Re_init):
    flow.Re.assign(Re)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    qB = solver.solve()

# Save steady state checkpoint for restart or post-processing
flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")

# Save visualization with velocity, pressure, and vorticity
vort = flow.vorticity()
try:
    # Try newer Firedrake API
    pvd = fd.VTKFile(f"{output_dir}/{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)
except AttributeError:
    # Fall back to older API
    pvd = fd.File(f"{output_dir}/{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {output_dir}/{Re}_steady.h5")
