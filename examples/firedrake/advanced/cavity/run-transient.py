#!/usr/bin/env python3
"""
Cavity Flow - Transient Simulation from Base Flow

Simulate unstable cavity flow starting from a steady base state.
Loads steady solution and adds perturbation to observe transition to turbulence.

Physical Behavior:
    - At Re=7500, cavity flow is highly unstable (well above critical Re)
    - Shear layer at cavity opening develops instabilities
    - Flow exhibits recirculation and complex vortex dynamics
    - Turbulent kinetic energy (TKE) measures fluctuation intensity

Usage:
    python run-transient.py              # Single-process execution
    mpirun -np 4 python run-transient.py  # Parallel execution

Outputs:
    - output/stats.dat: Time series of CFL, KE, TKE, memory
    - Console: Evolution of flow statistics

Prerequisites:
    - Requires steady state checkpoint from solve-steady.py
"""
import os
import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
pvd_out = None  # Set to file path to enable Paraview visualization
Re = 7500
restart = f"{output_dir}/{Re}_steady.h5"

# Create flow configuration and load steady state
flow = hgym.Cavity(Re=Re, restart=restart, use_HF_data_manager=False)

# Store base flow for computing TKE
# TKE measures energy in fluctuations relative to base flow
flow.qB.assign(flow.q)

# Add random perturbation to base flow to trigger turbulence
rng = fd.RandomGenerator(fd.PCG64(seed=1234))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-2)


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
  KE = 0.5 * fd.assemble(
      fd.inner(flow.u, flow.u) * fd.dx)  # Total kinetic energy
  TKE = flow.evaluate_objective(
  )  # Turbulent kinetic energy (fluctuation energy)
  CFL = flow.max_cfl(dt)  # CFL number for numerical stability monitoring
  mem_usage = psutil.virtual_memory().percent  # RAM usage
  return [CFL, KE, TKE, mem_usage]


# Configure logging callback to print and save time series data
print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.3e}\t\t TKE: {3:0.3e}\t\t Mem: {4:0.1f}"
)
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    # hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,  # Number of values returned by log_postprocess
        interval=1,  # Log every time step
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# Time integration parameters
Tf = 1.0  # Final time
method = "BDF"  # Backward Differentiation Formula (implicit time-stepping)
stabilization = "none"  # No additional spatial stabilization
dt = 1e-2  # Time step size

hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
)
