#!/usr/bin/env python3
"""
Cylinder Flow - Steady State Solver

Solve for steady-state flow around a circular cylinder at specified Reynolds number.
Uses Newton iteration with Reynolds ramping for high-Re flows.

Usage:
    python solve-steady.py

Physical setup:
    - Circular cylinder (radius = 0.5) in channel
    - Uniform inflow (U∞ = 1.0)
    - Re = 40 (subcritical, steady flow expected)
    - Outputs: Steady velocity, pressure, vorticity fields

Note: At Re=100, the steady state is linearly unstable (vortex shedding).
      This solver finds the unstable steady state, useful for stability analysis.
"""

import os
import firedrake as fd
import hydrogym.firedrake as hgym

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
mesh_resolution = "medium"
Re = 100  # Use Re < 47 for stable steady state, or Re=100 for unstable equilibrium

solver_parameters = {"snes_monitor": None}

# For Re > 50, ramp from lower Reynolds numbers for better convergence
if Re > 50:
  Re_init = [20, 40, 60, 80, Re]
else:
  Re_init = [Re]

flow = hgym.Cylinder(
    Re=Re_init[0],
    mesh=mesh_resolution,
    velocity_order=2,
    use_HF_data_manager=False)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

solver = hgym.NewtonSolver(
    flow,
    stabilization="gls",
    solver_parameters=solver_parameters,
)

# Reynolds ramping for convergence
for i, Re_val in enumerate(Re_init):
  flow.Re.assign(Re_val)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  qB = solver.solve()

# Save steady state
flow.save_checkpoint(f"{output_dir}/cylinder_Re{Re}_steady.h5")

# Compute and save force coefficients
CL, CD = flow.compute_forces()
hgym.print(f"Steady state forces: CL={CL:.6f}, CD={CD:.6f}")

# Save visualization
vort = flow.vorticity()
try:
  # Try newer Firedrake API
  pvd = fd.VTKFile(f"{output_dir}/cylinder_Re{Re}_steady.pvd")
  pvd.write(flow.u, flow.p, vort)
except AttributeError:
  # Fall back to older API
  pvd = fd.File(f"{output_dir}/cylinder_Re{Re}_steady.pvd")
  pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {output_dir}/cylinder_Re{Re}_steady.h5")
