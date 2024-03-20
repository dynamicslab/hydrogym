"""Solve the steady-state problem for the cylinder"""

import firedrake as fd
from cav_common import base_checkpoint, flow, stabilization

import hydrogym.firedrake as hgym

Re = 7500

solver_parameters = {"snes_monitor": None}

# Since this flow is at high Reynolds number we have to
#    ramp to get the steady state
Re_init = [500, 1000, 2000, 4000, Re]

dof = flow.mixed_space.dim()
hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

solver = hgym.NewtonSolver(
    flow,
    stabilization=stabilization,
    solver_parameters=solver_parameters,
)

for i, Re in enumerate(Re_init):
  flow.Re.assign(Re)
  hgym.print(f"Steady solve at Re={Re_init[i]}")
  qB = solver.solve()

# Save checkpoint
flow.q.assign(qB)
flow.save_checkpoint(base_checkpoint)
