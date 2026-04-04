#!/usr/bin/env python3
"""
Pinball Flow - Basic Transient Simulation

Simulate unsteady flow around three cylinders in triangular arrangement.
This demonstrates basic time integration with the Pinball flow configuration.

Physical Behavior:
    - Three cylinders create complex three-body wake interactions
    - At Re=30, flow may be steady or mildly unsteady
    - Higher Re values show complex periodic or chaotic wake dynamics
    - Force coefficients exhibit coupling between cylinders

Usage:
    python run-transient.py              # Single-process execution
    mpirun -np 4 python run-transient.py  # Parallel execution

Outputs:
    - coeffs.dat: Time series of lift coefficients for all three cylinders
    - Console: Time, CL for each cylinder, memory usage

Note: The controller function is defined but commented out in the integration call.
"""

import os

import numpy as np
import psutil

import hydrogym.firedrake as hgym

# Create output directory for checkpoints and data
output_dir = "."
pvd_out = None  # Set to a file path to enable Paraview visualization
restart = None  # Set to checkpoint path to continue from previous simulation
checkpoint = "checkpoint.h5"

# Create flow configuration with three cylinders in triangular arrangement
flow = hgym.Pinball(
    Re=30,  # Reynolds number (relatively low - may be steady or mildly unsteady)
    restart=restart,
    mesh="fine",
    use_HF_data_manager=False,  # Disable high-fidelity data manager for this example
)


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
    mem_usage = psutil.virtual_memory().percent  # RAM usage for monitoring
    obs = flow.get_observations()
    CL = obs[:3]  # Lift coefficients for all three cylinders
    return *CL, mem_usage


# Configure logging callback to print and save time series data
print_fmt = "t: {0:0.2f},\t\t CL[0]: {1:0.3f},\t\t CL[1]: {2:0.03f},\t\t CL[2]: {3:0.03f}\t\t Mem: {4:0.1f}"  # noqa: E501
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=4,  # Number of values returned by log_postprocess
    interval=1,  # Log every time step
    print_fmt=print_fmt,
    filename="coeffs.dat",  # Save time series to file
)

# Set up callbacks for logging and checkpointing
# Uncomment checkpoint callback below to save periodic restart files
# callbacks = [log, hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint)]
callbacks = [
    log,
]


# Controller function (currently not used - see commented line in integrate call)
def controller(t, obs):
    return np.array([0.0, 1.0, 1.0])  # Actuation inputs for three cylinders


# Time integration parameters
Tf = 1.0  # Final time
method = "BDF"  # Backward Differentiation Formula (implicit time-stepping)
stabilization = "none"  # No additional spatial stabilization
dt = 0.1  # Time step size

hgym.print("Beginning integration")
# Time integration with no control applied (controller commented out)
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
    # controller=controller,  # Uncomment to apply actuation
)
