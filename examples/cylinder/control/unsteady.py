import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'natural'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Cylinder(Re=50, h5_file='../output/checkpoint.h5')
# flow.solve_steady()  # Initialize with steady state

Tf = 300
dt = 1e-2

vort = fd.Function(flow.pressure_space, name='vort')
def compute_vort(flow):
    u, p = flow.u, flow.p
    vort.assign(fd.project(curl(u), flow.pressure_space))
    return (u, p, vort)

data = np.array([0, 0, 0], ndmin=2)
def forces(flow):
    global data
    CL, CD = flow.compute_forces(flow.q)
    return CL, CD

callbacks = [
    gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    gym.io.CheckpointCallback(interval=100, filename=chk_out),
    gym.io.LogCallback(postprocess=forces, nvals=2, interval=10,
            filename=f'{output_dir}/forces.dat',
            print_fmt='t:{0:0.2f}\t\t CL:{1:0.4f} \t\tCD::{2:0.4f}'
    )
]
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')

# # Test time-varying BC
# solver = gym.ts.IPCSTest(flow, dt)
# t_span = (0, Tf)
# solver.solve(t_span, callbacks=callbacks)
# flow.set_control(fd.Constant(0.5))
# solver.solve(t_span, callbacks=callbacks)