import os

import matplotlib.pyplot as plt
import numpy as np
import psutil  # For memory tracking
import scipy.io as sio

from hydrogym.firedrake.utils.pd import PDController
import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

mesh_resolution = "medium"

element_type = "p1p1"
velocity_order = 1
stabilization = "gls"

# Make sure to run the transient simulation first to generate the restart file
restart = f"{output_dir}/checkpoint_{mesh_resolution}_{element_type}.h5"
checkpoint = f"{output_dir}/pd_{mesh_resolution}_{element_type}.h5"


def compute_vort(flow):
  return (flow.u, flow.p, flow.vorticity())


def log_postprocess(flow):
  CL, CD = flow.get_observations()
  mem_usage = psutil.virtual_memory().available * 100 / psutil.virtual_memory(
  ).total
  mem_usage = psutil.virtual_memory().percent
  return CL, CD, mem_usage


callbacks = [
    # hgym.io.ParaviewCallback(
    #     interval=10, filename=f"{output_dir}/pd-control.pvd", postprocess=compute_vort
    # ),
    # hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=1,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f},\t\t RAM: {3:0.1f}%",
        filename=f"{output_dir}/pd-control.dat",
    ),
]

flow = hgym.RotaryCylinder(
    Re=100,
    mesh=mesh_resolution,
    restart=restart,
    callbacks=callbacks,
    velocity_order=velocity_order,
)

k = 2.0  # Base gain
theta = 4.0  # Phase angle
kp = k * np.cos(theta)
kd = k * np.sin(theta)

tf = 100.0
dt = 0.01
n_steps = int(tf // dt) + 2

pd_controller = PDController(
    kp,
    kd,
    dt,
    n_steps,
    filter_type="bilinear",
    N=20,
)


def controller(t, obs):
  # Turn on control halfway through
  if t < tf / 2:
    return 0.0
  return pd_controller(t, obs)


hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    controller=controller,
    stabilization=stabilization,
)

# Load results data
data = np.loadtxt(f"{output_dir}/pd-control.dat")

# Plot results
t = data[:, 0]
CL = data[:, 1]
CD = data[:, 2]

fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
axs[0].plot(t, CL)
axs[0].set_ylabel(r"$C_L$")
axs[1].grid()
axs[1].plot(t, CD)
axs[1].set_ylabel(r"$C_D$")
axs[1].grid()
axs[1].set_xlabel("Time $t$")
plt.show()
