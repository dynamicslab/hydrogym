# https://github.com/dynamicslab/hydrogym/blob/4c1c0e838147fff6cd3fd300294db14451ae120c/examples/cylinder/model-control/stability/stability.py
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from firedrake.pyplot import tripcolor, triplot
from scipy import sparse
from ufl import div, dot, dx, inner, nabla_grad

import hydrogym.firedrake as hgym

show_plots = False

flow = hgym.Cylinder(Re=100, mesh="medium")

# 1. Compute base flow
steady_solver = hgym.NewtonSolver(flow)
qB = steady_solver.solve()

# Check lift/drag
print(flow.compute_forces(qB))

if show_plots:
    vort = flow.vorticity(qB.subfunctions[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    tripcolor(vort, axes=ax, cmap="RdBu", vmin=-2, vmax=2)
    plt.show()


# 2. Construct linearized problem

# F = steady_solver.steady_form(qB)  # Nonlinear variational form
# J = fd.derivative(F, qB)  # Jacobian with automatic differentiation

uB = qB.subfunctions[0]
u, p = fd.TrialFunctions(flow.mixed_space)
v, s = fd.TestFunctions(flow.mixed_space)

# Linearized Navier-Stokes
A = -(
    inner(dot(u, nabla_grad(uB)), v) * dx
    + inner(dot(uB, nabla_grad(u)), v) * dx
    + inner(flow.sigma(u, p), flow.epsilon(v)) * dx
    + inner(div(u), s) * dx
)
M = inner(u, v) * dx
flow.linearize_bcs(mixed=True)
bcs = flow.collect_bcs()

problem = fd.LinearEigenproblem(A, M=M, bcs=bcs)

# # 3. Solve eigenvalue problem with SLEPc
# # NOTE: This doesn't work because the PETSc scalar type is real, so
# # specifying the shift-invert value as a complex number doesn't work.
# n_evals = 10
# solver_parameters = {
#     "eps_gen_non_hermitian": None,
#     "eps_target": "0.0+0.8i",
#     "eps_type": "krylovschur",
#     "eps_largest_real": True,
#     "st_type": "sinvert",
#     "st_pc_factor_shift_type": "NONZERO",
#     "eps_tol": 1e-10,
# }
# solver = fd.LinearEigensolver(
#     problem, n_evals=n_evals, solver_parameters=solver_parameters
# )
# n_evals = solver.solve()

# for i in range(n_evals):
#     print(solver.eigenvalue(i))


# if show_plots:
#     n_evals_plt = 5
#     fig, axs = plt.subplots(n_evals_plt, 2, figsize=(8, 12), sharex=True, sharey=True)

#     for i in range(n_evals_plt):
#         qr, qi = solver.eigenfunction(i)
#         ur = qr.subfunctions[0]
#         tripcolor(ur, axes=axs[i, 0], cmap="RdBu")
#         ui = qi.subfunctions[0]
#         tripcolor(ui, axes=axs[i, 1], cmap="RdBu")

#         print(ui.dat.data.min(), ui.dat.data.max())

#     plt.show()


# 3. Solve eigenvalue problem with SciPy (ARPACK)
# Will only work in serial... very slow

A = fd.assemble(A, bcs=bcs).petscmat  # Dynamics matrix
M = fd.assemble(M, bcs=bcs).petscmat  # Mass matrix


def petsc_to_scipy(petsc_mat):
    """Convert the PETSc matrix to a scipy CSR matrix"""
    indptr, indices, data = petsc_mat.getValuesCSR()
    scipy_mat = sparse.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
    return scipy_mat


def system_to_scipy(sys):
    """Convert the LTI system tuple (A, M, B) to scipy/numpy arrays"""
    A = petsc_to_scipy(sys[0])
    M = petsc_to_scipy(sys[1])
    if len(sys) == 2:
        return A, M
    B = np.vstack(sys[2]).T
    return A, M, B


A, M = system_to_scipy((A, M))

sigma_r = np.linspace(-4.0, 0.0, 10)
sigma_i = np.linspace(0.0, 4.0, 10)
n_evals = 10
tol = 1e-10
eigenvalues = {}
eigenfunctions = {}
for sr in sigma_r:
    for si in sigma_i:
        sigma = sr + 1j * si
        print(f"Computing eigenvalues for sigma = {sigma}")
        w, v = sparse.linalg.eigs(
            A, M=M, k=10, which="LR", sigma=sigma
        )  # , maxiter=1000)
        print(w)
        eigenvalues[sigma] = w
        eigenfunctions[sigma] = v

sio.savemat("eigenvalues.mat", eigenvalues)
sio.savemat("eigenfunctions.mat", eigenfunctions)
