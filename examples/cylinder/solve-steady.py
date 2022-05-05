import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl

import hydrogym as gym

output_dir = 'output'
flow = gym.flow.Cylinder()
# cyl.set_control(fd.Constant(0.5))
flow.solve_steady()

CL, CD = flow.compute_forces(flow.u, flow.p)
print(f'CL:{CL:08f} \t\tCD:{CD:08f}')

flow.save_checkpoint(f"{output_dir}/steady.h5")

vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/steady.pvd")
pvd.write(flow.u, flow.p, vort)