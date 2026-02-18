import firedrake as fd

import hydrogym.firedrake as hgym

output_dir = "output"
mesh_resolution = "fine"
Re = 7500

solver_parameters = {"snes_monitor": None}

# Since this flow is at high Reynolds number we have to
#    ramp to get the steady state
Re_init = [500, 1000, 2000, 4000, Re]
flow = hgym.Cavity(Re=Re_init[0], mesh=mesh_resolution, velocity_order=1)

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

flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")
vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort)
