#!/usr/bin/env python3
"""
Backward-Facing Step - Complete Unsteady Workflow

Two-stage simulation:
  1. Solve for steady-state base flow (Newton solver with Re ramping)
  2. Run transient to observe flow dynamics and potential instabilities

The step flow at Re=600 exhibits separation and may develop unsteady vortex shedding.

Usage:
    python unsteady.py

Physical setup:
    - Backward-facing step at Re=600
    - Flow develops from steady state
    - Long integration time (Tf=500) to capture dynamics
"""
import os

import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
mesh_resolution = "fine"
output_dir = f"./{Re}_{mesh_resolution}_output"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

# Time integration method and stabilization
method = "BDF"  # Backward Differentiation Formula
stabilization = "gls"  # Galerkin Least Squares for P1-P1 stability

# Create flow configuration with P1-P1 elements
flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=1,  # P1 velocity elements
    noise_amplitude=1.0,  # Inlet perturbation amplitude
    use_HF_data_manager=False,
)

# ========================================================================
# Stage 1: Solve steady state (base flow for TKE calculation)
# ========================================================================
hgym.print("=" * 70)
hgym.print("Stage 1: Solving for steady-state base flow")
hgym.print("=" * 70)

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
# Gradually increase Re from 100 to target value in steps of 100
Re_init = np.arange(100, Re + 100, 100, dtype=float)

for i, Re in enumerate(Re_init):
  flow.Re.assign(Re)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  solver = hgym.NewtonSolver(
      flow, solver_parameters=solver_parameters, stabilization=stabilization)
  flow.qB.assign(solver.solve())

# Save steady state checkpoint
flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")
hgym.print("Steady state computed - saving as base flow for TKE calculation")

# ========================================================================
# Stage 2: Transient simulation
# ========================================================================
hgym.print("\n" + "=" * 70)
hgym.print("Stage 2: Transient simulation")
hgym.print("=" * 70)

tf = 500  # Long integration to capture flow dynamics
dt = 1e-2


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


print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
)
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,  # Number of values returned by log_postprocess
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# Integrate in time - flow dynamics should develop
hgym.print("Starting time integration...\n")

# Time integration using BDF method with GLS stabilization
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
)

hgym.print("\n" + "=" * 70)
hgym.print("Simulation complete!")
hgym.print(f"Output saved to {output_dir}/")
hgym.print("=" * 70)
