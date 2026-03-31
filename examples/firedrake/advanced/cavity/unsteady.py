#!/usr/bin/env python3
"""
Cavity Flow - Complete Unsteady Workflow

Two-stage simulation:
  1. Solve for steady-state base flow (Newton solver with Re ramping)
  2. Perturb and run long transient to observe turbulence development

The cavity flow at Re=7500 exhibits shear-layer instability and turbulent dynamics.

Usage:
    python unsteady.py

Physical setup:
    - Open square cavity at Re=7500
    - Flow transitions from steady → turbulent
    - Long integration time (Tf=500) to capture statistical behavior
"""
import os
import firedrake as fd
import psutil

import hydrogym.firedrake as hgym

Re = 7500
mesh_resolution = "medium"  # Use 'medium' for balance of speed/accuracy
output_dir = f"./{Re}_{mesh_resolution}_output"
os.makedirs(output_dir, exist_ok=True)
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

# Create flow configuration with P1-P1 elements
flow = hgym.Cavity(Re=Re, mesh=mesh_resolution, use_HF_data_manager=False)

# ========================================================================
# Stage 1: Solve steady state (unstable equilibrium at Re=7500)
# ========================================================================
hgym.print("=" * 70)
hgym.print("Stage 1: Solving for steady-state base flow")
hgym.print("=" * 70)

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
# At high Re, direct solve often fails - start low and gradually increase
Re_init = [500, 1000, 2000, 4000, Re]

for i, Re in enumerate(Re_init):
    flow.Re.assign(Re)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    solver = hgym.NewtonSolver(flow, solver_parameters=solver_parameters)
    flow.qB.assign(solver.solve())

hgym.print("Steady state computed - saving as base flow for TKE calculation")

# ========================================================================
# Stage 2: Transient simulation from perturbed initial condition
# ========================================================================
hgym.print("\n" + "=" * 70)
hgym.print("Stage 2: Transient simulation with perturbation")
hgym.print("=" * 70)

Tf = 500  # Long integration to capture turbulent statistics
dt = 2.5e-4


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
  KE = 0.5 * fd.assemble(
      fd.inner(flow.u, flow.u) * fd.dx)  # Total kinetic energy
  TKE = flow.evaluate_objective(
  )  # Turbulent kinetic energy (fluctuation energy)
  CFL = flow.max_cfl(dt)  # CFL number for numerical stability monitoring
  mem_usage = psutil.virtual_memory().percent
  return [CFL, KE, TKE, mem_usage]


# Function to output velocity, pressure, and vorticity for visualization
def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


print_fmt = "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    # hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# Add random perturbation to trigger transition to turbulence
# This kicks the system away from the unstable steady state
hgym.print("Adding random perturbation to steady state...")
rng = fd.RandomGenerator(fd.PCG64(seed=1234))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-2)

# Integrate in time - turbulence should develop
hgym.print("Starting time integration...\n")
hgym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks)

hgym.print("\n" + "=" * 70)
hgym.print("Simulation complete!")
hgym.print(f"Output saved to {output_dir}/")
hgym.print("=" * 70)
