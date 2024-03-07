import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import tripcolor, triplot
from scipy import linalg, sparse
from ufl import cos, div, dot, dx, grad, inner

import hydrogym.firedrake as hgym

mesh = fd.UnitSquareMesh(10, 10)
V = fd.FunctionSpace(mesh, "CG", 1)
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
f = fd.Function(V)
x, y = fd.SpatialCoordinate(mesh)

f.interpolate((1 + 8 * np.pi * np.pi) * cos(x * np.pi * 2) * cos(y * np.pi * 2))

# Weak form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx

u_sol = fd.Function(V)

fd.solve(a == L, u_sol, solver_parameters={"ksp_type": "cg", "pc_type": "none"})

fig, ax = plt.subplots(figsize=(4, 4))
tripcolor(u_sol, axes=ax)
plt.show()


# Convert to SciPy
bcs = []
A = fd.assemble(-a, bcs=bcs).petscmat  # Dynamics matrix
M = fd.assemble(inner(u, v) * dx, bcs=bcs).petscmat  # Mass matrix


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
L = linalg.inv(M.todense()) @ A.todense()
evals, evecs = linalg.eig(L)

# Sort by descending real part (should be purely real)
sort_idx = np.argsort(-evals.real)
evals = evals[sort_idx]
evecs = evecs[:, sort_idx]
print(evals[:10].real)

# Plot the eigenmodes
n_evals_plt = 5
w = fd.Function(V)
fig, axs = plt.subplots(1, n_evals_plt, figsize=(12, 2), sharex=True, sharey=True)

for i in range(n_evals_plt):
    with w.dat.vec as vec:
        vec.array = evecs[:, i].real
    tripcolor(w, axes=axs[i], cmap="RdBu")
plt.show()
