from common import *

# Time step
Tf = 5.56
dt = Tf/500

# Tf = 300
# dt = 1e-2

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
    gym.io.CheckpointCallback(interval=100, filename=restart),
    gym.io.GenericCallback(callback=forces, interval=1),
    gym.io.SnapshotCallback(interval=5, filename=snap_file)
]
gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method='IPCS')