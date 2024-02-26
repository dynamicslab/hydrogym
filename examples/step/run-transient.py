import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

Re = 600
output_dir = "output"
mesh_resolution = "fine"
restart = f"{output_dir}/{Re}_steady.h5"

flow = hgym.Step(Re=Re, mesh=mesh_resolution, restart=restart)

# Store base flow for computing TKE
flow.qB.assign(flow.q)

tf = 1.0

# method = "IPCS"
# dt = 1e-3

method = "BDF"
dt = 1e-2


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = flow.evaluate_objective()
    CFL = flow.max_cfl(dt)
    mem_usage = psutil.virtual_memory().percent
    return [CFL, KE, TKE, mem_usage]


print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.6e}\t\t TKE: {3:0.6e}\t\t Mem: {4:0.1f}"
)
interval = int(1e-2 / dt)
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    # hgym.io.CheckpointCallback(interval=100, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=interval,
        filename=f"{output_dir}/{method}_stats.dat",
        print_fmt=print_fmt,
    ),
]


hgym.integrate(
    flow,
    t_span=(0, tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    eta=1.0,
)
