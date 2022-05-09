import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'controlled'
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"output/40_steady.h5"

Re = 40
# cyl = gym.flow.Cylinder(Re=Re)
# cyl.solve_steady()
# cyl.save_checkpoint(checkpoint)
cyl = gym.flow.Cylinder(Re=Re, h5_file=checkpoint)

# Time step
dt = 1e-2
Tf = 0.5

vort = fd.Function(cyl.pressure_space, name='vort')
def compute_vort(flow):
    u, p = flow.u, flow.p
    vort.assign(fd.project(curl(u), cyl.pressure_space))
    return (u, p, vort)

print_fmt = "t: {0:0.2f},\t\t CL:{1:0.3f},\t\t CD:{2:0.03f}"  # This will format the output
log = gym.io.LogCallback(
    postprocess=lambda flow: flow.collect_observations(),
    nvals=2,
    interval=1,
    print_fmt=print_fmt,
    filename=None
)

callbacks = [
    # gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    log
]

# callbacks = []
solver = gym.ts.IPCS(cyl, dt=dt)
# solver = gym.ts.IPCS_diff(cyl, dt=dt)

num_steps = int(Tf/dt)
for iter in range(num_steps):
    t = iter*dt
    y = cyl.collect_observations()
    omega = None
    if t > 0.2:
        omega = 0.1
    solver.step(iter, control=omega)
    for cb in callbacks:
        cb(iter, t, cyl)