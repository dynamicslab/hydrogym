import firedrake as fd

import hydrogym as gym

Re = 130
output_dir = "./output"
pvd_out = f"{output_dir}/solution.pvd"
checkpoint = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Pinball(Re=Re, mesh="coarse")

# Time step
Tf = 500
dt = 5e-3

h = fd.CellSize(flow.mesh)


def log_postprocess(flow):
    CL, CD = flow.compute_forces(flow.q)
    return [sum(CL), sum(CD)]


print_fmt = "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}"
callbacks = [
    gym.io.CheckpointCallback(interval=100, filename=checkpoint),
    gym.io.LogCallback(
        postprocess=log_postprocess,
        nvals=2,
        interval=10,
        filename=f"{output_dir}/stats.dat",
        print_fmt=print_fmt,
    ),
]

gym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method="IPCS")
