#!/usr/bin/env python3
"""
Cylinder Flow - PD Phase Sweep for Optimal Control Tuning

Systematically sweeps through PD controller phase angles to map control effectiveness.
This technique identifies the optimal phase relationship between actuation and
vortex shedding cycle for maximum suppression.

Control Parameterization:
    - Fixed gain magnitude: k = 0.1
    - Variable phase angle: θ ∈ [0, 2π]
    - PD coefficients: kp = k*cos(θ), kd = k*sin(θ)

Methodology:
    1. Start with developed vortex shedding (from restart file)
    2. Apply zero control for transient settling
    3. Sequentially apply 20 different phase angles
    4. Measure lift/drag response for each phase
    5. Analyze which phase best suppresses oscillations

Physical Insight:
    - Phase determines timing of actuation relative to vortex detachment
    - Optimal phase typically aligns actuation to counteract natural shedding
    - Results reveal flow's receptivity to different control strategies

Usage:
    python pd-phase-sweep.py

Prerequisites:
    - Must first run run-transient.py to generate checkpoint file

Outputs:
    - output/phase-sweep.dat: Time series with all phase angles
    - Use post-processing to extract performance metrics vs. phase

Application:
    - Controller tuning (find best kp/kd ratio)
    - System identification (frequency response)
    - Validating linear control theory predictions
"""
import os
import numpy as np
import psutil

from hydrogym.firedrake.utils.pd import PDController
import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

mesh_resolution = "medium"

# Finite element configuration
element_type = "p1p1"  # Equal-order velocity/pressure
velocity_order = 1  # Linear velocity approximation
stabilization = "none"

# IMPORTANT: Must run run-transient.py first to generate restart file
restart = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"
checkpoint = f"{output_dir}/pd_{mesh_resolution}_{element_type}.h5"


# Helper function for optional Paraview visualization
def compute_vort(flow):
  return (flow.u, flow.p, flow.vorticity())


# Extract force coefficients for performance analysis
def log_postprocess(flow):
  CL, CD = flow.get_observations()  # Lift and drag coefficients
  mem_usage = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
  mem_usage = psutil.virtual_memory().percent
  return CL, CD, mem_usage


# Configure logging callback
callbacks = [
    # Optional: Enable Paraview output for visualization
    # hgym.io.ParaviewCallback(
    #     interval=10, filename=f"{output_dir}/pd-control.pvd", postprocess=compute_vort
    # ),
    # Log force coefficients throughout entire sweep
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=1,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=f"{output_dir}/phase-sweep.dat",
    ),
]

# Create flow with rotary cylinder actuation
flow = hgym.RotaryCylinder(
    Re=100,
    mesh=mesh_resolution,
    restart=restart,  # Start from developed vortex shedding
    callbacks=callbacks,
    velocity_order=velocity_order,
)

# Phase sweep parameters
omega = 1 / 5.56  # Vortex shedding frequency (Hz) at Re=100 for this geometry
k = 0.1  # Fixed gain magnitude (small to avoid nonlinear saturation)
ctrl_time = 10 / omega  # Duration for each phase angle (≈10 shedding cycles)
n_phase = 20  # Number of phase angles to test
phasors = np.linspace(0.0, 2 * np.pi, n_phase)  # Phase angles from 0 to 2π

# Total simulation time: initial settling + n_phase intervals
tf = (n_phase + 1) * ctrl_time
dt = 1e-2  # Time step
n_steps = int(tf // dt) + 2

# Initialize PD controller with zero gains (will be updated in loop)
pd_controller = PDController(
    0.0,  # kp (will be set dynamically)
    0.0,  # kd (will be set dynamically)
    dt,
    n_steps,
    filter_type="bilinear",  # Tustin's method for derivative filter
    N=20,  # Filter rolloff
)


# Time-varying controller: switches phase angle every ctrl_time interval
def controller(t, obs):
  """
  Sequentially applies different PD phase angles.

  Timeline:
    - Interval 0 (0 < t < ctrl_time): Zero control (baseline)
    - Interval 1 (ctrl_time < t < 2*ctrl_time): Phase = 0°
    - Interval 2 (2*ctrl_time < t < 3*ctrl_time): Phase = 18°
    - ... and so on through 20 phase angles

  Args:
    t: Current time
    obs: Observation (lift coefficient CL)

  Returns:
    Control input (cylinder rotation rate)
  """
  # First interval: zero control for baseline measurement
  pd_controller.kp = 0.0
  pd_controller.kd = 0.0

  # Subsequent intervals: sweep through phase angles
  for j in range(n_phase):
    if t > (j + 1) * ctrl_time:
      # Convert phase angle to PD gains
      pd_controller.kp = k * np.cos(phasors[j])
      pd_controller.kd = k * np.sin(phasors[j])

  return pd_controller(t, obs)


# Run phase sweep experiment
# Expected output: Time series showing varying control effectiveness
# Post-process by extracting CL amplitude for each phase interval
hgym.print("Starting phase sweep experiment...")
hgym.print(f"Testing {n_phase} phase angles, {ctrl_time:.1f}s each")
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization=stabilization,
)
hgym.print(f"Phase sweep complete. Results saved to {output_dir}/phase-sweep.dat")
