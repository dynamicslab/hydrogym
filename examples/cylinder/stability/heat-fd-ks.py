"""Krylov-Schur eigenvalue analysis of the heat equation.

This uses finite differences, not Firedrake. It's just for the
sake of developing the Krylov-Schur code.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, sparse


def sparse_fd_matrix(n, dx):
    diagonals = np.array([-2 * np.ones(n), np.ones(n), np.ones(n)]) / dx**2
    offsets = np.array([0, -1, 1])
    # Zero Dirichlet boundary conditions: remove first and last rows and columns
    return sparse.dia_matrix((diagonals, offsets), shape=(n - 1, n - 1))


def arnoldi(A, v0, m=100, restart=None):
    n = len(v0)
    if restart is None:
        V = np.zeros((n, m))
        H = np.zeros((m, m))
        start_idx = 0
    else:
        V, H = restart
        start_idx = V.shape[1]

    if start_idx == m:
        print("Warning: restart array is full, no iterations performed.")

    f = v0
    for j in range(start_idx, m):
        beta = linalg.norm(f)
        H[j, j - 1] = beta
        v = f / beta
        V[:, j] = v
        w = A @ v
        H[: j + 1, j] = V[:, : j + 1].T @ w
        f = w - V[:, : j + 1] @ H[: j + 1, j]

    return V, H, v, beta


def eig_arnoldi(M, v0=None, m=100):
    n = M.shape[0]
    if v0 is None:
        v0 = np.random.rand(n)
    V, H, _, _ = arnoldi(M, v0, m)
    ritz_vals, ritz_vecs = linalg.eig(H)
    evecs = V @ ritz_vecs
    return ritz_vals, evecs


def eig_ks(M, v0=None, n_evals=10, m=100, tol=1e-10, delta=0.05, maxiter=None):
    n = M.shape[0]
    if v0 is None:
        v0 = np.random.rand(n)
    restart = None

    k = 0  # Iteration count (for logging)

    converged = []
    while len(converged) < n_evals:
        V, H, v0, beta = arnoldi(M, v0, m, restart=restart)

        # Check for convergence based on Arnoldi residuals
        ritz_vals, ritz_vecs = linalg.eig(H)
        converged = []
        for i in range(m):
            y = ritz_vecs[:, i]
            mu = ritz_vals[i]
            res = abs(beta * y[-1])
            if res < tol:
                converged.append((mu, y))

        print(f"Iteration {k}: {len(converged)} converged")

        # If enough eigenvalues have converged, return them and exit
        if len(converged) >= n_evals or (maxiter is not None and k >= maxiter):
            if maxiter is not None and k >= maxiter:
                print(f"Maximum number of iterations reached ({maxiter}). Exiting.")
            evals = np.zeros(len(converged), dtype=np.complex128)
            evecs = np.zeros((n, len(converged)), dtype=np.complex128)
            for i, (mu, y) in enumerate(converged):
                evals[i] = mu
                evecs[:, i] = V @ y
            return evals, evecs

        # Schur decomposition
        Q, S, p = linalg.schur(H, sort=lambda x: abs(x) > 1.0 - delta)

        # Keep the "wanted" part of the Schur form. These will be the eigenvalues
        # that are closest to the unit circle (least stable).
        Hp = S[:p, :p]

        # Re-order the Krylov basis
        Vp = V @ Q[:, :p]

        # Restart with the wanted part of the Krylov basis
        restart = (Vp, Hp)


if __name__ == "__main__":
    # 1. Set up the finite difference matrix for the heat equation Dxx
    n = 200
    L = 1.0
    dt = 0.01  # Time step for time propagator
    A = sparse_fd_matrix(n, L / n)

    # 2. Time propagator
    # Here we'll use the exact solution expm(A*t), so we need to
    # use the dense matrix.  Otherwise the Krylov methods can work fine
    # with sparse matrices or matrix-free operators
    M = linalg.expm(A.toarray() * dt)

    # 3. Eigenvalues of the finite difference matrix (dense eig)
    evals, evecs = linalg.eigh(A.toarray())
    sort_idx = np.argsort(-evals)
    dense_evals = evals[sort_idx]
    dense_evecs = evecs[:, sort_idx]

    # 4. Eigenvalues of the finite difference matrix (Arnoldi)
    arn_evals_dt, arn_evecs = eig_arnoldi(M)
    sort_idx = np.argsort(-abs(arn_evals_dt))
    arn_evals_dt = arn_evals_dt[sort_idx]
    arn_evecs = arn_evecs[:, sort_idx]
    arn_evals = np.log(arn_evals_dt) / dt

    # 5. Eigenvalues of the finite difference matrix (Krylov-Schur)
    ks_evals_dt, ks_evecs = eig_ks(M, delta=0.99)
    sort_idx = np.argsort(-abs(ks_evals_dt))
    ks_evals_dt = ks_evals_dt[sort_idx]
    ks_evecs = ks_evecs[:, sort_idx]
    ks_evals = np.log(ks_evals_dt) / dt

    # Print the leading eigenvalues for each method
    n_print = 5
    print(dense_evals.shape)
    print(f"Dense eigenvalues: {dense_evals[:n_print]}")
    arn_err = abs(dense_evals[:n_print] - arn_evals[:n_print].real)
    print(f"Arnoldi eigenvalues: {arn_evals[:n_print].real}")
    print(f"    error: {arn_err}")
    ks_err = abs(dense_evals[:n_print] - ks_evals[:n_print].real)
    print(f"Krylov-Schur eigenvalues: {ks_evals[:n_print].real}")
    print(f"    error: {ks_err}")
