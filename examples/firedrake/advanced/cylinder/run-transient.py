#!/usr/bin/env python3
"""
Cylinder Flow - Basic Transient Simulation

Simulate uncontrolled vortex shedding behind a circular cylinder at Re=100.
This demonstrates basic time integration with the RotaryCylinder flow configuration.

Physical Behavior:
    - At Re=100, flow is supercritical (beyond Hopf bifurcation at Re≈47)
    - Vortex shedding develops naturally from random initial conditions
    - Expected behavior: Oscillating lift (CL) and drag (CD) coefficients
    - Strouhal number St = f*D/U ≈ 0.165 (shedding frequency)

Usage:
    python run-transient.py              # Single-process execution
    mpirun -np 4 python run-transient.py  # Parallel execution

Outputs:
    - output/checkpoint_*.h5: Flow state for restart/post-processing
    - Console: Time, lift coefficient, drag coefficient, memory usage

Note: This uses RotaryCylinder (with tangential actuation capability),
      but no control is applied (zero actuation input).
"""
import os
import psutil
import hydrogym.firedrake as hgym

# Create output directory for checkpoints
output_dir = "output"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

pvd_out = None  # Set to a file path to enable Paraview visualization
restart = None  # Set to checkpoint path to continue from previous simulation
mesh_resolution = "medium"

# Finite element configuration
element_type = "p1p1"  # P1-P1 elements (equal-order velocity/pressure)
velocity_order = 1  # Linear velocity approximation
stabilization = "none"  # P1-P1 requires stabilization in general, but works here

checkpoint = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"

# Create flow configuration with rotary cylinder actuation
# (tangential velocity boundary condition on cylinder surface)
flow = hgym.RotaryCylinder(
    Re=100,  # Reynolds number (supercritical - vortex shedding expected)
    restart=restart,
    mesh=mesh_resolution,
    velocity_order=velocity_order,
)

# Time integration parameters
tf = 300.0  # Final time 
dt = 0.01  # Time step size (adjust for CFL condition)


# Custom function to extract quantities of interest at each time step
def log_postprocess(flow):
  mem_usage = psutil.virtual_memory().percent  # RAM usage for monitoring
  CL, CD = flow.get_observations()  # Lift and drag coefficients
  return CL, CD, mem_usage


# Configure logging callback to print and save time series data
print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=3,  # Number of values returned by log_postprocess
    interval=1,  # Log every time step
    print_fmt=print_fmt,
    filename=None,  # Set to path to save data file (e.g., "coeffs.dat")
)

# Checkpoint every 100 time units for restart capability
interval = int(100 / dt)
callbacks = [
    log,
    hgym.utils.io.CheckpointCallback(interval=interval, filename=checkpoint),
]

hgym.print("Beginning integration")
# Time integration using BDF (Backward Differentiation Formula) method
# - Implicit time-stepping for stability
# - No control applied (controller=None by default)
# - Callbacks executed at each time step
hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    stabilization=stabilization,  # Additional spatial stabilization if needed
)
