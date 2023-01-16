import firedrake as fd
import numpy as np
from ufl import sqrt

import hydrogym as hgym

Re = 7500
output_dir = f"{Re}_output"
checkpoint = f"{output_dir}/checkpoint.h5"
pvd_out = f"{output_dir}/open_loop.pvd"

# First we have to ramp up the Reynolds number to get the steady state
Re_init = [500, 1000, 2000, 4000, Re]
flow = hgym.flow.Cavity(Re=Re_init[0], mesh="fine")
hgym.print(f"Steady solve at Re={Re_init[0]}")
qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

for (i, Re) in enumerate(Re_init[1:]):
    flow.Re.assign(Re)
    hgym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

# Time step
dt = 1e-4
Tf = 20.0


def compute_vort(flow):
    u, p = flow.u, flow.p
    return (u, p, flow.vorticity())


h = fd.CellSize(flow.mesh)


def log_postprocess(flow):
    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    CFL = (
        fd.project(
            dt * sqrt(flow.u.sub(0) ** 2 + flow.u.sub(1) ** 2) / h, flow.pressure_space
        )
        .vector()
        .max()
    )
    return [CFL, KE]


print_fmt = "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}"
callbacks = [
    hgym.io.ParaviewCallback(interval=1000, filename=pvd_out, postprocess=compute_vort),
    hgym.io.CheckpointCallback(interval=100, filename=checkpoint),
    hgym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=2,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

# gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks)

solver = hgym.ts.IPCS(flow, dt=dt)
n_steps = int(Tf // dt)

for i in range(n_steps):
    t = dt * i
    solver.step(i, control=np.sin(t))
    for cb in callbacks:
        cb(i, t, flow)
