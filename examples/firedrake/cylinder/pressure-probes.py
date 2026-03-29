"""
Simulate the flow around the cylinder with evenly spaced surface pressure probes.
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

velocity_order = 1
stabilization = "gls"
mesh = "medium"
method = "BDF"

# Configure pressure probes - evenly spaced around the cylinder
n_probes = 8
R = 0.5
probes = [(R * np.cos(theta), R * np.sin(theta))
          for theta in np.linspace(0, 2 * np.pi, n_probes, endpoint=False)]

flow = hgym.Cylinder(
    Re=100,
    mesh=mesh,
    velocity_order=velocity_order,
    observation_type="vorticity_probes",
    probes=probes,
)

# Time step
Tf = 10.0
dt = 0.1


def log_postprocess(flow):
  mem_usage = psutil.virtual_memory().percent
  p = flow.get_observations()
  return *p, mem_usage


# Set up the callback
print_fmt = "t: {0:0.2f},\t\t p[4]: {5}"  # This will format the output
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=n_probes + 1,
    interval=1,
    print_fmt=print_fmt,
    filename=data_file,
)

callbacks = [
    log,
    # hgym.utils.io.CheckpointCallback(interval=10, filename=checkpoint),
]


def controller(t, y):
  return [flow.MAX_CONTROL]


hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
    # controller=controller,
)

data = np.loadtxt(data_file)
t = data[:, 0]
p = data[:, 1:n_probes + 1]

plt.figure(figsize=(7, 2))
plt.plot(t, p)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.show()
