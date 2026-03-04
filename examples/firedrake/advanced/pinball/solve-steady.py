#!/usr/bin/env python3
"""
Pinball Flow - Steady State Solver

Solve for steady-state flow around three cylinders in triangular arrangement.
Uses Newton iteration with Reynolds ramping for convergence.

Usage:
    python solve-steady.py

Physical setup:
    - Three cylinders in equilateral triangle configuration
    - Uniform inflow (U∞ = 1.0)
    - Re = 100 (default)
    - Complex wake structure with three-body interaction

Note: Pinball flow exhibits rich dynamics even at moderate Reynolds numbers.
      Steady state may be unstable, useful for stability analysis.
"""

import firedrake as fd
import hydrogym.firedrake as hgym

output_dir = "output"
mesh_resolution = "medium"
Re = 80  # Use Re < 100 for better steady convergence

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for better convergence
Re_init = [40, 60, Re]

flow = hgym.Pinball(Re=Re_init[0], mesh=mesh_resolution, velocity_order=2)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

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
flow.save_checkpoint(f"{output_dir}/pinball_Re{Re}_steady.h5")

# Compute and save force coefficients for each cylinder
CL1, CL2, CL3 = flow.compute_forces()
hgym.print(f"Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL1:.6f}")
hgym.print(f"  Cylinder 2: CL={CL2:.6f}")
hgym.print(f"  Cylinder 3: CL={CL3:.6f}")

# Save visualization
vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/pinball_Re{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {output_dir}/pinball_Re{Re}_steady.h5")
