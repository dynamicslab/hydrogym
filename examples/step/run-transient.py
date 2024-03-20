import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
mesh_resolution = "m5"
output_dir = f"./{Re}_{mesh_resolution}_output"
restart = f"{output_dir}/steady.h5"
checkpoint = f"{output_dir}/checkpoint.h5"

flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    restart=restart,
    velocity_order=1,
    noise_amplitude=1.0,
    noise_seed=0,  # For reproducibility across meshes
)

# Store base flow for computing TKE
flow.qB.assign(flow.q)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

tf = 1000.0
method = "BDF"
stabilization = "gls"
dt = 0.01


def log_postprocess(flow):
  KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
  TKE = flow.evaluate_objective()
  CFL = flow.max_cfl(dt)
  mem_usage = psutil.virtual_memory().percent
  return [CFL, KE, TKE, mem_usage]


print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
)
interval = max(1, int(1e-1 / dt))
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=interval,
        filename=f"{output_dir}/stats.dat",
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
)
