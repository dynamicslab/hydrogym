import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import logging

import hydrogym as gym

Re = 80
output_dir = f'{Re}_output'
flow = gym.flow.Pinball(Re=Re, mesh_name='fine')
flow.solve_steady()

CL, CD = flow.compute_forces()
gym.print([(L, D) for (L, D) in zip(CL, CD)])

flow.save_checkpoint(f"{output_dir}/steady.h5")

vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/steady.pvd")
pvd.write(flow.u, flow.p, vort)