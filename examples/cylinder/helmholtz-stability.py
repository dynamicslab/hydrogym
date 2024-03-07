# Remove before merge!!

import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor
from ufl import dx, grad, inner

mesh = fd.UnitSquareMesh(10, 10)
V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)
A = (inner(grad(u), grad(v)) + inner(u, v)) * dx
problem = fd.LinearEigenproblem(A)
n_evals = 10
solver_parameters = {
    "eps_largest_real": None,
    "eps_gen_hermitian": None,
}
solver = fd.LinearEigensolver(
    problem, n_evals=n_evals, solver_parameters=solver_parameters
)
n_evals = solver.solve()


n_evals_plt = 5
fig, axs = plt.subplots(n_evals_plt, 2, figsize=(4, 12), sharex=True, sharey=True)

for i in range(n_evals_plt):
    print(solver.eigenvalue(i))
    ur, ui = solver.eigenfunction(i)
    tripcolor(ur, axes=axs[i, 0])
    tripcolor(ui, axes=axs[i, 1])

plt.show()
