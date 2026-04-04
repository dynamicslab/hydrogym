#!/usr/bin/env python3
"""
Pinball Flow - Complete Unsteady Workflow

Two-stage simulation:
  1. Solve for steady-state base flow (Newton solver)
  2. Perturb and run transient to observe wake dynamics

The pinball configuration creates a complex wake with three-body interactions,
exhibiting rich dynamics including wake switching and chaotic behavior.

Usage:
    python unsteady.py

Physical setup:
    - Three cylinders in triangular arrangement at Re=100
    - Flow transitions from steady → complex periodic/chaotic wake
    - Wake can switch between symmetric and asymmetric modes
"""

import firedrake as fd
import psutil

import hydrogym.firedrake as hgym

Re = 100
mesh_resolution = "medium"
output_dir = f"./pinball_Re{Re}_{mesh_resolution}_output"
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

# Create flow configuration with Taylor-Hood elements (P2-P1)
flow = hgym.Pinball(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=2,  # P2 velocity, P1 pressure (Taylor-Hood)
    use_HF_data_manager=False,  # Disable high-fidelity data manager
)

# ========================================================================
# Stage 1: Solve steady state
# ========================================================================
hgym.print("=" * 70)
hgym.print("Stage 1: Solving for steady-state base flow")
hgym.print("=" * 70)

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
# Start from lower Re and gradually increase to target value
Re_init = [40, 60, 80, Re]

for i, Re_val in enumerate(Re_init):
    flow.Re.assign(Re_val)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    solver = hgym.NewtonSolver(
        flow,
        stabilization="none",  # Taylor-Hood (P2-P1) is inf-sup stable
        solver_parameters=solver_parameters,
    )
    qB = solver.solve()

# Save steady state as base flow for reference
flow.qB = flow.q.copy(deepcopy=True)
CL, CD = flow.compute_forces()
hgym.print("Steady state forces:")
hgym.print(f"  Cylinder 1: CL={CL[0]:.6f}, CD={CD[0]:.6f}")
hgym.print(f"  Cylinder 2: CL={CL[1]:.6f}, CD={CD[1]:.6f}")
hgym.print(f"  Cylinder 3: CL={CL[2]:.6f}, CD={CD[2]:.6f}")

# ========================================================================
# Stage 2: Transient simulation from perturbed initial condition
# ========================================================================
hgym.print("\n" + "=" * 70)
hgym.print("Stage 2: Transient simulation with perturbation")
hgym.print("=" * 70)

Tf = 200  # Long integration to capture complex wake dynamics
dt = 0.01


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
    obs = flow.get_observations()
    CL1, CL2, CL3 = obs[:3]  # Extract lift coefficients (first 3 values)
    CFL = flow.max_cfl(dt)  # CFL number for numerical stability monitoring
    mem_usage = psutil.virtual_memory().percent
    return [CFL, CL1, CL2, CL3, mem_usage]


# Function to output velocity, pressure, and vorticity for visualization
def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


print_fmt = "t: {0:0.2f}\t\tCFL: {1:0.2f}\t\tCL1: {2:0.3f}\t\tCL2: {3:0.3f}\t\tCL3: {4:0.3f}\t\tMem: {5:0.1f}"
callbacks = [
    hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=500, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=5,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# Add small random perturbation to trigger instability
# This kicks the system away from the (potentially unstable) steady state
hgym.print("Adding random perturbation to steady state...")
rng = fd.RandomGenerator(fd.PCG64(seed=42))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-3)

# Integrate in time - complex wake dynamics should develop
hgym.print("Starting time integration...\n")

# Time integration using BDF (Backward Differentiation Formula) method
# - GLS stabilization for robustness with P2-P1 elements
# - Callbacks save visualization, checkpoints, and time series data
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    stabilization="gls",
)

hgym.print("\n" + "=" * 70)
hgym.print("Simulation complete!")
hgym.print(f"Output saved to {output_dir}/")
hgym.print("=" * 70)
