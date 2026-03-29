import numpy as np
import psutil

import hydrogym.firedrake as hgym

output_dir = "."
pvd_out = None
restart = None
checkpoint = "checkpoint.h5"

flow = hgym.Pinball(
    Re=30,
    restart=restart,
    mesh="fine",
)


def log_postprocess(flow):
  mem_usage = psutil.virtual_memory().percent
  obs = flow.get_observations()
  CL = obs[:3]
  return *CL, mem_usage


# Set up the callback
print_fmt = "t: {0:0.2f},\t\t CL[0]: {1:0.3f},\t\t CL[1]: {2:0.03f},\t\t CL[2]: {3:0.03f}\t\t Mem: {4:0.1f}"  # noqa: E501
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=4,
    interval=1,
    print_fmt=print_fmt,
    filename="coeffs.dat",
)

# callbacks = [log, hgym.utils.io.CheckpointCallback(interval=100, filename=checkpoint)]
callbacks = [
    log,
]


def controller(t, obs):
  return np.array([0.0, 1.0, 1.0])


# Simulation config
Tf = 1.0
method = "BDF"
stabilization = "gls"
dt = 0.1

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
