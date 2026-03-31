#!/usr/bin/env python3
"""
Cylinder Flow - Complete Unsteady Workflow

Two-stage simulation:
  1. Solve for steady-state base flow (Newton solver)
  2. Perturb and run transient to observe vortex shedding

This demonstrates a typical workflow for flow instability studies:
  - Find unstable steady state
  - Add small perturbation
  - Watch instability grow into limit cycle (vortex shedding)

Usage:
    python unsteady.py

Physical setup:
    - Circular cylinder at Re=100
    - Flow transitions from steady → periodic vortex shedding
    - Expected Strouhal number St ≈ 0.165
"""

import os
import firedrake as fd
import psutil
import hydrogym.firedrake as hgym

Re = 100
mesh_resolution = "medium"
output_dir = f"./cylinder_Re{Re}_{mesh_resolution}_output"
os.makedirs(output_dir, exist_ok=True)
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

flow = hgym.Cylinder(
    Re=Re, mesh=mesh_resolution, velocity_order=2, use_HF_data_manager=False)

# ========================================================================
# Stage 1: Solve steady state (unstable equilibrium at Re=100)
# ========================================================================
hgym.print("=" * 70)
hgym.print("Stage 1: Solving for steady-state base flow")
hgym.print("=" * 70)

solver_parameters = {"snes_monitor": None}

# Reynolds ramping for convergence
Re_init = [40, 60, 80, Re]

for i, Re_val in enumerate(Re_init):
  flow.Re.assign(Re_val)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  solver = hgym.NewtonSolver(
      flow,
      stabilization="none",  # Taylor-Hood (P2-P1) is inf-sup stable
      solver_parameters=solver_parameters)
  qB = solver.solve()

# Save steady state as base flow
flow.qB = flow.q.copy(deepcopy=True)
CL_steady, CD_steady = flow.compute_forces()
hgym.print(f"Steady state forces: CL={CL_steady:.6f}, CD={CD_steady:.6f}")

# ========================================================================
# Stage 2: Transient simulation from perturbed initial condition
# ========================================================================
hgym.print("\n" + "=" * 70)
hgym.print("Stage 2: Transient simulation with perturbation")
hgym.print("=" * 70)

Tf = 200  # Integrate for ~16 vortex shedding cycles
dt = 0.01


def log_postprocess(flow):
  CL, CD = flow.get_observations()
  CFL = flow.max_cfl(dt)
  mem_usage = psutil.virtual_memory().percent
  return [CFL, CL, CD, mem_usage]


def compute_vort(flow):
  return (flow.u, flow.p, flow.vorticity())


print_fmt = "t: {0:0.2f}\t\tCFL: {1:0.2f}\t\tCL: {2:0.4f}\t\tCD: {3:0.4f}\t\tMem: {4:0.1f}"
callbacks = [
    hgym.io.ParaviewCallback(
        interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=500, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# Add small random perturbation to trigger instability
hgym.print("Adding random perturbation to steady state...")
rng = fd.RandomGenerator(fd.PCG64(seed=42))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-3)

# Integrate in time - vortex shedding should develop
hgym.print("Starting time integration...\n")

# Try without stabilization first (Taylor-Hood is inf-sup stable)
# This uses the fast FGMRES + Schur complement solver
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    stabilization="none",  # Change to "supg" if you see oscillations
)

hgym.print("\n" + "=" * 70)
hgym.print("Simulation complete!")
hgym.print(f"Output saved to {output_dir}/")
hgym.print("=" * 70)
