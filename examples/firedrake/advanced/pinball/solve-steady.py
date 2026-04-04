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

import os

import firedrake as fd

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_resolution = "medium"
Re = 80  # Use Re < 100 for better steady convergence

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for better convergence
# Start from lower Re and gradually increase to target value
Re_init = [40, 60, Re]

# Create flow configuration with Taylor-Hood elements (P2-P1)
flow = hgym.Pinball(
    Re=Re_init[0],
    mesh=mesh_resolution,
    velocity_order=2,  # P2 velocity, P1 pressure (Taylor-Hood)
    use_HF_data_manager=False,  # Disable high-fidelity data manager
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

# Set up Newton solver for nonlinear steady-state problem
solver = hgym.NewtonSolver(
    flow,
    stabilization="none",  # Taylor-Hood (P2-P1) is inf-sup stable
    solver_parameters=solver_parameters,
)

# Reynolds ramping for convergence
for i, Re_val in enumerate(Re_init):
    flow.Re.assign(Re_val)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    qB = solver.solve()

# Save steady state checkpoint for restart or post-processing
flow.save_checkpoint(f"{output_dir}/pinball_Re{Re}_steady.h5")

# Compute and save force coefficients for each cylinder
# Returns arrays with [CL1, CL2, CL3] and [CD1, CD2, CD3]
CL, CD = flow.compute_forces()
hgym.print("Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL[0]:.6f}, CD={CD[0]:.6f}")
hgym.print(f"  Cylinder 2: CL={CL[1]:.6f}, CD={CD[1]:.6f}")
hgym.print(f"  Cylinder 3: CL={CL[2]:.6f}, CD={CD[2]:.6f}")

# Save visualization
vort = flow.vorticity()
try:
    # Try newer Firedrake API
    pvd = fd.VTKFile(f"{output_dir}/pinball_Re{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)
except AttributeError:
    # Fall back to older API
    pvd = fd.File(f"{output_dir}/pinball_Re{Re}_steady.pvd")
    pvd.write(flow.u, flow.p, vort)

hgym.print(f"Steady state saved to {output_dir}/pinball_Re{Re}_steady.h5")
