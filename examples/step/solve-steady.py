import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

output_dir = "output"
mesh_resolution = "fine"
Re = 600
checkpoint_prefix = f"{output_dir}/{Re}_steady"

solver_parameters = {"snes_monitor": None}

# Since this flow is at high Reynolds number we have to
#    ramp to get the steady state
Re_init = np.arange(100, Re + 100, 100, dtype=float)
flow = hgym.Step(Re=Re, mesh=mesh_resolution)

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

for i, Re in enumerate(Re_init):
    flow.Re.assign(Re)
    hgym.print(f"Steady solve at Re={Re_init[i]}")
    solver = hgym.NewtonSolver(flow, solver_parameters=solver_parameters)
    qB = solver.solve()

flow.save_checkpoint(f"{checkpoint_prefix}.h5")
vort = flow.vorticity()
pvd = fd.File(f"{checkpoint_prefix}.pvd")
pvd.write(flow.u, flow.p, vort)
