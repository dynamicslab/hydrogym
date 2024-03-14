import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import tripcolor, triplot
from scipy import linalg
from ufl import dx, grad, inner

# Heat equation in finite elements:
# A = k * inner(grad(u), grad(v)) * dx
# M = inner(u, v) * dx
# So the system solve A @ u1 = M @ u0 is equivalent to solving:
# inner(u0, v) * dx = k * inner(grad(u), grad(v))
# With spectral shift of sigma: A <- A - sigma * M
# Solving (A - sigma * M) @ u1 = M @ u0 is equivalent to solving:
# inner(u0, v) * dx = k * inner(grad(u), grad(v)) - sigma * inner(u, v) * dx


def solve_matrix_pencil(A, M, u0, sigma=0.0):
    # Solve the matrix pencil A @ f = M @ v for f
    # This is equivalent to the "inverse iteration" f = (A^{-1} @ M) @ v
    # Here `v` is a Firedrake function and `M` and `A` are functions that should
    # produce
    fn_space = v0.function_space()
    u, v = A.arguments()
    a = A - sigma * M(u)
    L = M(u0)
    bcs = [fd.DirichletBC(fn_space, 0, "on_boundary")]
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    u_sol = fd.Function(fn_space)
    fd.solve(a == L, u_sol, bcs=bcs, solver_parameters=solver_parameters)
    return u_sol


def arnoldi(A, M, v0, inner_product, m=100, sigma=0.0):

    V = [v0.copy(deepcopy=True).assign(0.0) for _ in range(m + 1)]
    H = np.zeros((m, m))
    start_idx = 1
    V[0].assign(v0)

    for j in range(start_idx, m + 1):
        # Apply iteration M @ f = A @ v by solving the linear system
        f = solve_matrix_pencil(A, M, V[j - 1], sigma=sigma)

        # Gram-Schmidt
        h = np.zeros(j)
        for i in range(j):
            h[i] = inner_product(V[i], f)
            f.assign(f - V[i] * h[i])

        # Fill in the upper Hessenberg matrix
        H[:j, j - 1] = h

        # Norm of the orthogonal residual
        beta = np.sqrt(inner_product(f, f))

        if j < m:
            H[j, j - 1] = beta

        # Normalize the orthogonal residual for use as a new
        # Krylov basis vector
        V[j].assign(f / beta)

    return V, H, beta


def eig_arnoldi(A, M, v0, inner_product, m=100, sigma=0.0, sort=None):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-abs(x))

    V, H, _ = arnoldi(A, M, v0, inner_product, m, sigma)
    ritz_vals, ritz_vecs = linalg.eig(H)

    sort_idx = sort(ritz_vals)
    ritz_vals = ritz_vals[sort_idx]
    ritz_vecs = ritz_vecs[:, sort_idx]

    evals = 1 / (ritz_vals + sigma)  # Undo spectral shift

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(m)]
    evecs_imag = [fd.Function(fn_space) for _ in range(m)]
    for i in range(m):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(m)))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(m)))
    return evals, evecs_real, evecs_imag


if __name__ == "__main__":

    mesh = fd.UnitSquareMesh(10, 10)
    fn_space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    k = 0.1

    u = fd.TrialFunction(fn_space)
    v = fd.TestFunction(fn_space)

    def inner_product(u, v):
        return fd.assemble(inner(u, v) * dx)

    def M(u):
        return inner(u, v) * dx

    A = -k * inner(grad(u), grad(v)) * dx

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(fn_space)
    alpha = np.sqrt(inner_product(v0, v0))
    v0.assign(v0 / alpha)

    arn_evals, arn_evecs_real, arn_evecs_imag = eig_arnoldi(
        A, M, v0, inner_product, m=20, sigma=0.0
    )

    n_print = 5
    print(f"Arnoldi eigenvalues: {arn_evals[:n_print].real}")
    # print(f"Krylov-Schur eigenvalues: {ks_evals[:n_print].real}")

    # # Plot the eigenmodes
    # n_evals_plt = 5
    # fig, axs = plt.subplots(1, n_evals_plt, figsize=(12, 2), sharex=True, sharey=True)

    # for i in range(n_evals_plt):
    #     tripcolor(ks_evecs_real[i], axes=axs[i], cmap="RdBu")
    # plt.show()
