import firedrake as fd
from ufl import sqrt

import hydrogym as gym

Re = 600
output_dir = "./output"
pvd_out = f"{output_dir}/solution.pvd"
restart = "./checkpoint.h5"
restart = None
checkpoint = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Step(Re=Re, h5_file=restart, mesh="coarse")

# Time step
Tf = 500
dt = 1e-3

h = fd.CellSize(flow.mesh)


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    TKE = KE - flow.BASE_KE
    CFL = (
        fd.project(
            dt * sqrt(flow.u.sub(0) ** 2 + flow.u.sub(1) ** 2) / h, flow.pressure_space
        )
        .vector()
        .max()
    )
    return [CFL, TKE]


print_fmt = "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t TKE: {2:0.12e}"
callbacks = [
    gym.io.CheckpointCallback(interval=10000, filename=checkpoint),
    gym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=2,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
    gym.io.ParaviewCallback(
        postprocess=lambda flow: (flow.u, flow.p, flow.vorticity()),
        filename=pvd_out,
        interval=1000,
    ),
]

gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method="IPCS", eta=1.0)
