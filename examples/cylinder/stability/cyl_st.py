from functools import partial

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from cyl_common import base_checkpoint, evec_checkpoint, flow, inner_product
from firedrake.pyplot import tripcolor, triplot
from scipy import linalg
from ufl import dx, grad, inner

import hydrogym.firedrake as hgym


def arnoldi(solve, v0, inner_product, m=100):
    """Solve Arnoldi iteration for the inverse iteration of a matrix pencil."""
    V = [v0.copy(deepcopy=True).assign(0.0) for _ in range(m + 1)]
    H = np.zeros((m, m))
    start_idx = 1
    V[0].assign(v0)

    f = v0.copy(deepcopy=True)
    for j in range(start_idx, m + 1):
        # Apply iteration A @ f = M @ v
        # This is the iteration f = (A^{-1} @ M) @ v
        # which is inverse to M @ f = A @ v
        # Stores the result in `f`
        solve(f, V[j - 1])

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

        # Normalize the orthogonal residual for use as a new Krylov vector
        V[j].assign(f / beta)

        # DEBUG: Print the eigenvalues
        # TODO: Make this optional. Also use `sorted_eig` and better formatting
        if True:
            hgym.print(f"*** Arnoldi iteration {j} ***")
            ritz_vals, ritz_vecs = linalg.eig(H[:j, :j])
            evals = 1 / ritz_vals
            sort_idx = np.argsort(-evals.real)
            evals = evals[sort_idx]
            ritz_vecs = ritz_vecs[:, sort_idx]
            hgym.print(f"Eigvals: {evals[:20]}")
            res = abs(beta * ritz_vecs[-1, :20])
            hgym.print(f"Residuals: {res}")

    return V, H, beta


def sorted_eig(H, sort="None", inverse=True):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-x.real)

    evals, evecs = linalg.eig(H)
    if inverse:
        evals = 1 / evals  # Undo spectral shift
    sort_idx = sort(evals)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    return evals, evecs


def eig_arnoldi(solve, v0, inner_product, m=100, sort=None):
    V, H, _ = arnoldi(solve, v0, inner_product, m)
    evals, ritz_vecs = sorted_eig(H, sort=sort, inverse=True)

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(m)]
    evecs_imag = [fd.Function(fn_space) for _ in range(m)]
    for i in range(m):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(m)))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(m)))
    return evals, evecs_real, evecs_imag


if __name__ == "__main__":
    flow.load_checkpoint(base_checkpoint)
    qB = flow.q.copy(deepcopy=True)
    fn_space = flow.mixed_space
    flow.linearize_bcs()
    bcs = flow.collect_bcs()

    # MUMPS sparse direct LU solver
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    (uB, pB) = qB.subfunctions
    q_trial = fd.TrialFunction(fn_space)
    q_test = fd.TestFunction(fn_space)
    (v, s) = fd.split(q_test)

    # Linear form expressing the _LHS_ of the Navier-Stokes without time derivative
    # For a steady solution this is F(qB) = 0.
    # TODO: Make this a standalone function - could be used in NewtonSolver and transient
    newton_solver = hgym.NewtonSolver(flow)
    F = newton_solver.steady_form(qB, q_test=q_test)
    # The Jacobian of F is the bilinear form J(qB, q_test) = dF/dq(qB) @ q_test
    J = -fd.derivative(F, qB, q_trial)

    def M(q):
        u = q.subfunctions[0]
        return inner(u, v) * dx

    # TODO: May need both real and imaginary solves for complex sigma
    # The shifted bilinear form should be `A = (J - sigma * M)`
    sigma = 0.0
    A = J

    def solve(v1, v0):
        """Solve the matrix pencil A @ v1 = M @ v0 for v1.

        This is equivalent to the "inverse iteration" v1 = (A^{-1} @ M) @ v0

        Stores the result in `v1`
        """
        fd.solve(A == M(v0), v1, bcs=bcs, solver_parameters=solver_parameters)

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(fn_space)
    alpha = np.sqrt(inner_product(v0, v0))
    v0.assign(v0 / alpha)

    rvals, evecs_real, evecs_imag = eig_arnoldi(solve, v0, inner_product, m=100)
    # Undo spectral shift
    evals = sigma + rvals

    # Sort by decreasing real part
    sort_idx = np.argsort(-evals.real)
    evals = evals[sort_idx]
    evecs_real = [evecs_real[i] for i in sort_idx]
    evecs_imag = [evecs_imag[i] for i in sort_idx]

    n_save = 32
    print(f"Arnoldi eigenvalues: {evals[:n_save]}")

    # Save checkpoints
    chk_dir, chk_file = evec_checkpoint.split("/")
    chk_path = "/".join([chk_dir, f"st_{chk_file}"])
    np.save("/".join([chk_dir, "st_evals"]), evals[:n_save])

    with fd.CheckpointFile(chk_path, "w") as chk:
        for i in range(n_save):
            evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(evecs_real[i])
            evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(evecs_imag[i])
