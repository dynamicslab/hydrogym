#!/usr/bin/env python3
"""
Cylinder Flow - Surface Pressure Measurements with Probes

Demonstrates using point probes to measure flow quantities at specific locations.
In this case, vorticity is measured at 8 evenly spaced points around the cylinder
surface.

Applications:
    - Sensor placement studies
    - Reduced-order modeling from limited measurements
    - Validating control strategies with sparse sensing
    - Mimicking experimental pressure tap arrays

Physical Setup:
    - 8 probes evenly distributed around cylinder (θ = 0, 45°, 90°, ...)
    - Probe locations: (R*cos(θ), R*sin(θ)) where R = 0.5
    - Measurements taken every time step

Usage:
    python pressure-probes.py

Outputs:
    - output/pressure.dat: Time series of probe measurements
    - Plot: Time evolution of all probe signals
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import psutil

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

data_file = f"{output_dir}/pressure.dat"

# Numerical configuration
velocity_order = 1  # P1 elements (linear velocity approximation)
stabilization = "none"  # No additional stabilization
mesh = "medium"
method = "BDF"  # Backward Differentiation Formula (implicit time-stepping)

# Configure probe array - evenly spaced around the cylinder surface
n_probes = 8  # Number of measurement points
R = 0.5  # Cylinder radius
probes = [(R * np.cos(theta), R * np.sin(theta))
          for theta in np.linspace(0, 2 * np.pi, n_probes, endpoint=False)]

# Create flow with custom observation points
flow = hgym.Cylinder(
    Re=100,
    mesh=mesh,
    velocity_order=velocity_order,
    observation_type="pressure_probes",
    probes=probes,  # List of (x, y) coordinates
    use_HF_data_manager=False,
)

# Time integration parameters
Tf = 10.0  # Final time (long enough to see several vortex shedding cycles)
dt = 0.1  # Time step


# Extract probe measurements and memory usage at each time step
def log_postprocess(flow):
  mem_usage = psutil.virtual_memory().percent
  p = flow.get_observations()  # Returns array of probe values [p1, p2, ..., p8]
  return *p, mem_usage  # Unpack probe values + memory


# Configure data logging
# Print format shows time and one probe value (probe 4) as example
print_fmt = "t: {0:0.2f},\t\t p[4]: {5}"
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=n_probes + 1,  # 8 probes + 1 memory value
    interval=1,  # Log every time step
    print_fmt=print_fmt,
    filename=data_file,  # Save all probe data to file
)

callbacks = [
    log,
    # Optional: Add checkpoint callback for restart capability
    # hgym.utils.io.CheckpointCallback(interval=10, filename=checkpoint),
]


# Optional controller function (currently commented out in integrate call)
def controller(t, y):
  return [flow.MAX_CONTROL]  # Apply maximum control input


hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
    # controller=controller,  # Uncomment to apply control
)

# Load and visualize probe measurements
data = np.loadtxt(data_file)
t = data[:, 0]  # Time column
p = data[:, 1:n_probes + 1]  # Probe measurements (columns 1-8)

# Plot all probe time series
plt.figure(figsize=(7, 2))
plt.plot(t, p)  # Each column plots as separate line
plt.grid()
plt.xlabel("Time")
plt.ylabel("Vorticity at Probes")  # Update label to match actual measurement
plt.title(f"Surface Measurements from {n_probes} Probes")
plt.show()
