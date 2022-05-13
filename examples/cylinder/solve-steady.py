# import numpy as np
# from petsc4py import PETSc
# PETSc.ScalarType = np.float64

import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl

import hydrogym as gym

mesh = 'noack'
output_dir = 'output'
flow = gym.flow.Cylinder(mesh_name=mesh, Re=100)
flow.set_control(fd.Constant(0.1))
flow.solve_steady()

CL, CD = flow.compute_forces()
# print(f'CL:{CL:08f} \t\tCD:{CD:08f}')
gym.print((CL, CD))

# flow.save_checkpoint(f"{output_dir}/{mesh}-steady.h5", write_mesh=True)

# vort = flow.vorticity()
# pvd = fd.File(f"{output_dir}/steady.pvd")
# pvd.write(flow.u, flow.p, vort)