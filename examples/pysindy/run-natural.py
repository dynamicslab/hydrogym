import firedrake as fd
import psutil

import hydrogym.firedrake as hgym

output_dir = "output"
checkpoint_dir = "checkpoints"
pvd_out = None
restart = None
checkpoint = f"{checkpoint_dir}/checkpoint_coarse.h5"

rng = fd.RandomGenerator(fd.PCG64(seed=1234))

flow = hgym.Cylinder(Re=100, restart=restart, mesh="coarse")
solver = hgym.NewtonSolver(flow, solver_parameters={"snes_monitor": None})
solver.solve()

flow.save_checkpoint(f"{checkpoint_dir}/steady.h5")

# Random perturbation to base flow for initialization
flow.q += rng.normal(flow.mixed_space, 0.0, 1e-4)

# Time step
Tf = 300
dt = 1e-2


def log_postprocess(flow):
    mem_usage = psutil.virtual_memory().percent
    CL, CD = flow.get_observations()
    return CL, CD, mem_usage


# Set up the callback
print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"
log = hgym.utils.io.LogCallback(
    postprocess=log_postprocess,
    nvals=3,
    interval=1,
    print_fmt=print_fmt,
    filename=f"{output_dir}/results.dat",
)

callbacks = [log, hgym.utils.io.CheckpointCallback(interval=1000, filename=checkpoint)]

hgym.print("Beginning integration")
hgym.integrate(flow, t_span=(0, Tf), dt=dt, callbacks=callbacks, method="IPCS")
