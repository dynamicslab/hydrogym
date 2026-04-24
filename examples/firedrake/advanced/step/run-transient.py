#!/usr/bin/env python3
"""
Backward-Facing Step - Transient Simulation from Base Flow

Simulate separated flow starting from a steady base state.
Loads steady solution and adds perturbation to observe flow dynamics.

Physical Behavior:
    - At Re=600, step flow exhibits separation and reattachment
    - Recirculation zone forms behind step
    - Flow may develop unsteady vortex shedding
    - Turbulent kinetic energy (TKE) measures fluctuation intensity

Usage:
    python run-transient.py              # Single-process execution
    mpirun -np 4 python run-transient.py  # Parallel execution

Outputs:
    - output/stats.dat: Time series of CFL, KE, TKE, memory
    - output/checkpoint.h5: Periodic checkpoints for restart
    - Console: Evolution of flow statistics

Prerequisites:
    - Requires steady state checkpoint from solve-steady.py
"""

import os

import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
mesh_resolution = "fine"
restart = f"{output_dir}/{Re}_steady.h5"
checkpoint = f"{output_dir}/checkpoint.h5"

# Create flow configuration and load steady state
flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    restart=restart,  # Load from steady state checkpoint
    velocity_order=1,  # P1 velocity elements
    noise_amplitude=1.0,  # Amplitude of inlet perturbation
    use_HF_data_manager=False,
)

# Store base flow for computing TKE (turbulent kinetic energy)
# TKE measures energy in fluctuations relative to base flow
flow.qB.assign(flow.q)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof / fd.COMM_WORLD.size)}")

# Time integration parameters
tf = 1000.0  # Final time - long integration to capture dynamics
method = "BDF"  # Backward Differentiation Formula (implicit time-stepping)
stabilization = "gls"  # Galerkin Least Squares stabilization for P1-P1
dt = 0.01  # Time step size


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)  # Total kinetic energy
    TKE = flow.evaluate_objective()  # Turbulent kinetic energy (fluctuation energy)
    CFL = flow.max_cfl(dt)  # CFL number for numerical stability monitoring
    mem_usage = psutil.virtual_memory().percent  # RAM usage
    return [CFL, KE, TKE, mem_usage]


# Configure logging callback to print and save time series data
print_fmt = "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
interval = max(1, int(1e-1 / dt))  # Log every 0.1 time units
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,  # Number of values returned by log_postprocess
        interval=interval,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

hgym.print("Beginning integration")
# Time integration with GLS stabilization for P1-P1 elements
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
)
