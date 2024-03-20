import os

import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

mesh_resolution = "fine"
Re = 600
output_dir = f"./{Re}_{mesh_resolution}_output"

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

checkpoint_prefix = f"{output_dir}/steady"

solver_parameters = {"snes_monitor": None}

# Since this flow is at high Reynolds number we have to
#    ramp to get the steady state
Re_init = np.arange(100, Re + 100, 100, dtype=float)
flow = hgym.Step(
    Re=Re,
    mesh=mesh_resolution,
    velocity_order=1,
)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

solver = hgym.NewtonSolver(
    flow,
    stabilization="gls",
    solver_parameters=solver_parameters,
)

for i, Re in enumerate(Re_init):
  flow.Re.assign(Re)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  qB = solver.solve()

flow.save_checkpoint(f"{checkpoint_prefix}.h5")
vort = flow.vorticity()
pvd = fd.File(f"{checkpoint_prefix}.pvd")
pvd.write(flow.u, flow.p, vort)
