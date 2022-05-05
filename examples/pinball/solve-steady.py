import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import logging

import hydrogym as gym

output_dir = 'output'
flow = gym.flow.Pinball(Re=80)
flow.solve_steady()

CL, CD = flow.compute_forces(flow.u, flow.p)
print([(L, D) for (L, D) in zip(CL, CD)])

flow.save_checkpoint(f"{output_dir}/steady.h5")

vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/steady.pvd")
pvd.write(flow.u, flow.p, vort)