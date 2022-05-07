import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'output'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"
snap_dir = 'snapshots'

flow = gym.flow.Cylinder(Re=100, h5_file=chk_out)
# flow.solve_steady()  # Initialize with steady state

# Time step
Tf = 5.56
dt = Tf/500

vort = fd.Function(flow.pressure_space, name='vort')
def compute_vort(flow):
    u, p = flow.u, flow.p
    vort.assign(fd.project(curl(u), flow.pressure_space))
    return (u, p, vort)

data = np.array([0, 0, 0], ndmin=2)
def forces(iter, t, flow):
    global data
    CL, CD = flow.compute_forces(flow.q)
    # if fd.COMM_WORLD.rank == 0:
    #     data = np.append(data, np.array([t, CL, CD], ndmin=2), axis=0)
    #     np.savetxt(f'{output_dir}/coeffs.dat', data)
    gym.print(f't:{t:08f}\t\t CL:{CL:08f} \t\tCD::{CD:08f}')

callbacks = [
    # gym.io.ParaviewCallback(interval=10, filename=pvd_out, postprocess=compute_vort),
    # gym.io.CheckpointCallback(interval=100, filename=chk_out),
    gym.io.GenericCallback(callback=forces, interval=1),
    gym.io.SnapshotCallback(interval=5, output_dir=snap_dir)
]
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')

# # Test time-varying BC
# solver = gym.ts.IPCSTest(flow, dt)
# t_span = (0, Tf)
# solver.solve(t_span, callbacks=callbacks)
# flow.set_control(fd.Constant(0.5))
# solver.solve(t_span, callbacks=callbacks)