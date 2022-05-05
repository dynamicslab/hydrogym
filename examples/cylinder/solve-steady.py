import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl

import hydrogym as gym

output_dir = 'output'
cyl = gym.flow.Cylinder()
cyl.set_control(fd.Constant(0.5))
cyl.solve_steady()

CL, CD = cyl.compute_forces(cyl.u, cyl.p)
print(f'CL:{CL:08f} \t\tCD:{CD:08f}')

cyl.save_checkpoint(f"{output_dir}/steady.h5")