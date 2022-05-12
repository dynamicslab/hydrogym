import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl

import hydrogym as gym

output_dir = 'output'
Re = 4000
Re_init = 2000

h5_file = f'{output_dir}/{Re_init}_steady.h5'
flow = gym.flow.Cavity(Re=Re, h5_file=h5_file)
solver_parameters={"snes_monitor": None,
                   "ksp_type": "gmres",
                   "mat_type": "aij",
                   "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"}

flow.solve_steady(solver_parameters=solver_parameters)
flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")
vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort)