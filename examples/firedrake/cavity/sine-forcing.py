import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

Re = 7500
mesh_resolution = "coarse"
output_dir = f"{Re}_{mesh_resolution}_output"
restart = f"{Re}_{mesh_resolution}_output/checkpoint.h5"
pvd_out = f"{output_dir}/open_loop.pvd"

# Time step
dt = 1e-4
Tf = 20.0


def compute_vort(flow):
  u, p = flow.u, flow.p
  return (u, p, flow.vorticity())


def log_postprocess(flow):
  KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
  TKE = flow.evaluate_objective()
  CFL = flow.max_cfl(dt)
  return [CFL, KE, TKE]


env_config = {
    "flow":
        hgym.Cavity,
    "flow_config": {
        "restart": restart,
    },
    "solver":
        hgym.IPCS,
    "solver_config": {
        "dt": dt,
    },
    "callbacks": [
        hgym.io.ParaviewCallback(
            interval=1000, filename=pvd_out, postprocess=compute_vort),
        hgym.io.LogCallback(
            postprocess=log_postprocess,
            nvals=3,
            interval=10,
            filename=f"{output_dir}/stats_ol.dat",
            print_fmt="t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.12e}\t\t TKE: {3:0.12e}",
        ),
    ],
}
env = hgym.FlowEnv(env_config)
num_steps = int(Tf // dt)
for i in range(num_steps):
  t = dt * i
  env.step(np.sin(t))
