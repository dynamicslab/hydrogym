import firedrake as fd

import hydrogym as gym

output_dir = "output"
Re = 600

# solver_parameters = {
#     "snes_monitor": None,
#     "ksp_type": "preonly",
#     "mat_type": "aij",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }

# solver_parameters = {
#     "snes_monitor": None,
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


Re_init = [100, 200, 300, 400, 500, Re]
flow = gym.flow.Step(Re=Re_init[0])

for (i, R) in enumerate(Re_init):
    flow.Re.assign(R)
    gym.print(f"Steady solve at Re={R}")
    qB = flow.solve_steady(solver_parameters=solver_parameters)

    KE = 0.5 * fd.assemble(fd.inner(flow.u, flow.u) * fd.dx)
    gym.print(f"KE at Re={R}: {KE}")

# flow.set_control(1.0)
vort = flow.vorticity()

# Compute shear stress
u, p = qB.split()
tau = fd.project((1 / Re) * u[0].dx(1), flow.pressure_space)
tau.rename("tau")

flow.save_checkpoint(f"{output_dir}/{R}_steady.h5")
pvd = fd.File(f"{output_dir}/{Re}_steady.pvd")
pvd.write(flow.u, flow.p, vort, tau)
