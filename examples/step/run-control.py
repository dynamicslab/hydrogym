import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
output_dir = "output"
mesh_resolution = "medium_mesh_sensors"
checkpoint = f"{output_dir}/checkpoint.h5"

flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=1,
    observation_type="reattachment",
    noise_amplitude=0.5,
)

def controller(t, obs):
  return flow.MAX_CONTROL if t > 30.0 else 0.0

flow.qB.assign(flow.q)

tf = 100.0
method = "BDF"
stabilization = "gls"
dt = 0.01

def compute_vort(flow):
  return (flow.u, flow.p, flow.vorticity())

def log_postprocess(flow):
  XR = flow.reattachment_length()[0]
  TKE = flow.evaluate_objective()
  CFL = flow.max_cfl(dt)
  mem_usage = psutil.virtual_memory().percent 
  return [CFL, XR, TKE, mem_usage]

print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t XR: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
)

interval = max(1, int(1e-1 / dt))
callbacks = [
    hgym.io.ParaviewCallback(interval=100, filename=f"{output_dir}/step-control.pvd", postprocess=compute_vort),
    #hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=interval,
        filename=f"{output_dir}/step_control.dat",
        print_fmt=print_fmt,
    ),
]

hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
    controller=controller,
)
