import firedrake as fd
import numpy as np
import psutil

import hydrogym.firedrake as hgym

output_dir = "output"
pvd_out = None
Re = 7500
restart = f"{output_dir}/{Re}_steady.h5"

flow = hgym.Cavity(Re=Re, restart=restart)

# Store base flow for computing TKE
flow.qB.assign(flow.q)

# Random perturbation to base flow for initialization
rng = fd.RandomGenerator(fd.PCG64(seed=1234))
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-2)


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = flow.evaluate_objective()
    CFL = flow.max_cfl(dt)
    mem_usage = psutil.virtual_memory().percent
    return [CFL, KE, TKE, mem_usage]


print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.3e}\t\t TKE: {3:0.3e}\t\t Mem: {4:0.1f}"
)
callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    # hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=4,
        interval=1,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]


# End time of the simulation
Tf = 1.0
method = "BDF"  # Time-stepping method
stabilization = "gls"  # Stabilization method
dt = 1e-2


hgym.print("Beginning integration")
hgym.integrate(
    flow,
    t_span=(0, Tf),
    dt=dt,
    callbacks=callbacks,
    method=method,
    stabilization=stabilization,
)
