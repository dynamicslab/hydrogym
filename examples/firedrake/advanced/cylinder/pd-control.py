#!/usr/bin/env python3
"""
Cylinder Flow - PD Control for Vortex Shedding Suppression

Demonstrates Proportional-Derivative (PD) control to suppress vortex shedding
using rotary actuation on the cylinder surface. The controller activates
halfway through the simulation to show the transition from uncontrolled
(oscillating) to controlled (stabilized) flow.

Control Strategy:
    - Sensor: Lift coefficient CL (measures vortex shedding amplitude)
    - Actuator: Cylinder rotation (tangential surface velocity)
    - Controller: PD control with filtered derivative
        u(t) = kp*CL(t) + kd*dCL/dt

Physical Interpretation:
    - Proportional term (kp): Reacts to current lift magnitude
    - Derivative term (kd): Anticipates lift rate of change
    - Phase angle θ = atan(kd/kp): Determines control timing relative to vortex cycle

Tuning Parameters:
    - Gain magnitude k: Overall control strength
    - Phase angle θ: Timing of actuation relative to shedding
    - Filter parameter N: Smooths noisy derivative estimates

Usage:
    python pd-control.py

Prerequisites:
    - Must first run run-transient.py to generate checkpoint file

Outputs:
    - output/pd-control.dat: Time series of CL, CD during control
    - Plot: Lift and drag showing oscillation → stabilization transition
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import psutil  # For memory tracking
import scipy.io as sio

import hydrogym.firedrake as hgym
from hydrogym.firedrake.utils.pd import PDController

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mesh_resolution = "medium"

# Finite element configuration
element_type = "p1p1"  # Equal-order velocity/pressure elements
velocity_order = 1  # Linear velocity approximation
stabilization = "none"  # No additional stabilization needed

# IMPORTANT: Must run run-transient.py first to generate the restart file
# The restart file provides a developed vortex shedding state as initial condition
restart = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"
checkpoint = f"{output_dir}/pd_{mesh_resolution}_{element_type}.h5"


# Helper function for Paraview visualization output
def compute_vort(flow):
    return (flow.u, flow.p, flow.vorticity())


# Extract force coefficients at each time step for logging
def log_postprocess(flow):
    CL, CD = flow.get_observations()  # Lift and drag coefficients
    mem_usage = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    mem_usage = psutil.virtual_memory().percent  # RAM usage percentage
    return CL, CD, mem_usage


# Configure output callbacks
callbacks = [
    # Optional: Enable Paraview output to visualize flow fields
    # hgym.io.ParaviewCallback(
    #     interval=10, filename=f"{output_dir}/pd-control.pvd", postprocess=compute_vort
    # ),
    # Optional: Save checkpoints for restart capability
    # hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint),
    # Log force coefficients to file and console
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,  # CL, CD, memory
        interval=1,  # Log every time step
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=f"{output_dir}/pd-control.dat",
    ),
]

# Create flow with rotary cylinder actuation
flow = hgym.RotaryCylinder(
    Re=100,  # Reynolds number (supercritical regime)
    mesh=mesh_resolution,
    restart=restart,  # Load developed vortex shedding state
    callbacks=callbacks,
    velocity_order=velocity_order,
    use_HF_data_manager=False,
)

# PD controller tuning parameters
# These values are expressed in polar form: (magnitude, phase angle)
k = 2.0  # Gain magnitude (overall control strength)
theta = 4.0  # Phase angle in radians (timing of actuation)

# Convert polar form to Cartesian (kp, kd) coefficients
# This parameterization makes it easier to sweep phase angles
kp = k * np.cos(theta)  # Proportional gain
kd = k * np.sin(theta)  # Derivative gain

# Time integration setup
tf = 100.0  # Final time (run long enough to see stabilization)
dt = 0.01  # Time step
n_steps = int(tf // dt) + 2  # Total number of time steps

# Create PD controller with filtered derivative
# Filter prevents noise amplification from numerical differentiation
pd_controller = PDController(
    kp,  # Proportional gain
    kd,  # Derivative gain
    dt,  # Time step for derivative approximation
    n_steps,  # Pre-allocate arrays for efficiency
    filter_type="bilinear",  # Tustin's method for discrete-time filter
    N=20,  # Filter rolloff parameter (higher = more filtering)
)


# Controller wrapper: enable control halfway through simulation
# This creates a clear before/after comparison in the results
def controller(t, obs):
    # Phase 1: No control - observe natural vortex shedding
    if t < tf / 2:
        return 0.0  # Zero actuation
    # Phase 2: PD control active - suppress vortex shedding
    return pd_controller(t, obs)  # obs = lift coefficient CL


# Run time integration with control
# Expected behavior:
#   - First half (t < 50): Oscillating CL, CD (uncontrolled vortex shedding)
#   - Second half (t > 50): CL → 0, CD → steady value (shedding suppressed)
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,  # Apply PD control
    stabilization=stabilization,
)

# ========================================================================
# Post-processing: Visualize control performance
# ========================================================================

# Load time series data
data = np.loadtxt(f"{output_dir}/pd-control.dat")

t = data[:, 0]  # Time
CL = data[:, 1]  # Lift coefficient
CD = data[:, 2]  # Drag coefficient

# Create two-panel plot showing lift and drag evolution
fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

# Lift coefficient (should decay to ~0 after control activates)
axs[0].plot(t, CL)
axs[0].set_ylabel(r"$C_L$")
axs[0].grid()
axs[0].axvline(tf / 2, color="red", linestyle="--", label="Control ON")
axs[0].legend()

# Drag coefficient (should stabilize after control activates)
axs[1].plot(t, CD)
axs[1].set_ylabel(r"$C_D$")
axs[1].grid()
axs[1].axvline(tf / 2, color="red", linestyle="--")
axs[1].set_xlabel("Time $t$")

plt.tight_layout()
plt.show()
