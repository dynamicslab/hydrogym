import firedrake as fd

import hydrogym as gym

output_dir = "output"
Re = 4000


# solver_parameters = {
#     "snes_monitor": None,
#     "ksp_type": "preonly",
#     "mat_type": "aij",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }


# solver_parameters = {
#     "snes_monitor": None,
#     "ksp_monitor": None,
#     "ksp_type": "fgmres",
#     "pc_type": "fieldsplit",
#     "pc_fieldsplit_type": "schur",
#     "pc_fieldsplit_detect_saddle_point": None,
#     "pc_fieldsplit_schur_fact_type": "full",
#     "pc_fieldsplit_schur_precondition": "selfp",
#     "fieldsplit_0_ksp_type": "preonly",
#     "fieldsplit_1_ksp_type": "gmres",
#     "fieldsplit_1_pc_type": "hypre",
#     "fieldsplit_1_hypre_type": "boomeramg",
# }


# solver_parameters = {
#     "snes_monitor": None,
#     "ksp_monitor": None,
#     "mat_type": "matfree",
#     "ksp_type": "fgmres",
#     "pc_type": "fieldsplit",
#     "pc_fieldsplit_type": "schur",
#     "pc_fieldsplit_schur_fact_type": "diag",
#     "fieldsplit_0_ksp_type": "preonly",
#     "fieldsplit_0_pc_type": "python",
#     "fieldsplit_0_pc_python_type": "firedrake.MassInvPC",
#     "fieldsplit_1_Mp_ksp_type": "preonly",
#     "fieldsplit_1_Mp_pc_type": "lu",
# }

solver_parameters = {"snes_monitor": None}

# First we have to ramp up the Reynolds number to get the steady state
Re_init = [500, 1000, 2000, 4000, Re]
flow = gym.flow.Cavity(Re=Re_init[0], mesh="fine")
gym.print(f"Steady solve at Re={Re_init[0]}")
qB = flow.solve_steady(solver_parameters=solver_parameters)

for (i, Re) in enumerate(Re_init[1:]):
    flow.Re.assign(Re)
    gym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters=solver_parameters)

flow.save_checkpoint(f"{output_dir}/{Re}_steady.h5")
vort = flow.vorticity()
pvd = fd.File(f"{output_dir}/{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort)
