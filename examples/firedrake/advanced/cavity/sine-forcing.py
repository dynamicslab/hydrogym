#!/usr/bin/env python3
"""
Cavity Flow - Sinusoidal Forcing Control

Demonstrates open-loop periodic actuation on cavity flow.
Applies sinusoidal actuation input and observes flow response.

Usage:
    python sine-forcing.py

Physical setup:
    - Cavity flow at Re=7500
    - Periodic actuation: u(t) = sin(t)
    - Monitors kinetic energy and TKE response

Prerequisites:
    - Requires checkpoint from previous simulation (unsteady.py or run-transient.py)
"""

import os

import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

Re = 7500
mesh_resolution = "coarse"
output_dir = f"{Re}_{mesh_resolution}_output"
os.makedirs(output_dir, exist_ok=True)
restart = f"{Re}_{mesh_resolution}_output/checkpoint.h5"
pvd_out = f"{output_dir}/open_loop.pvd"

# Time integration parameters
dt = 1e-4  # Small time step for temporal accuracy
Tf = 20.0  # Final time


# Function to output velocity, pressure, and vorticity for visualization
def compute_vort(flow):
    u, p = flow.u, flow.p
    return (u, p, flow.vorticity())


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)  # Total kinetic energy
    TKE = flow.evaluate_objective()  # Turbulent kinetic energy
    CFL = flow.max_cfl(dt)  # CFL number for numerical stability monitoring
    return [CFL, KE, TKE]


# Configure flow environment with callbacks
# This uses the FlowEnv interface for RL-style step() API
env_config = {
    "flow": hgym.Cavity,
    "flow_config": {
        "restart": restart,  # Load from checkpoint
        "use_HF_data_manager": False,
    },
    "solver": hgym.SemiImplicitBDF,  # BDF time-stepping scheme
    "solver_config": {
        "dt": dt,
    },
    "callbacks": [
        hgym.io.ParaviewCallback(interval=1000, filename=pvd_out, postprocess=compute_vort),
        hgym.io.LogCallback(
            postprocess=log_postprocess,
            nvals=3,
            interval=10,
            filename=f"{output_dir}/stats_ol.dat",
            print_fmt="t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}\t\t TKE: {3:0.12e}",
        ),
    ],
}

# Create environment and run with sinusoidal actuation
env = hgym.FlowEnv(env_config)
num_steps = int(Tf // dt)

hgym.print(f"Running {num_steps} time steps with sinusoidal forcing")
for i in range(num_steps):
    t = dt * i
    env.step(np.sin(t))  # Apply sinusoidal actuation

hgym.print(f"Simulation complete! Output saved to {output_dir}/")
