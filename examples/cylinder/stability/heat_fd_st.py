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


def arnoldi(A, M, v0, m=100, inverse=True):
    n = len(v0)
    V = np.zeros((n, m + 1))
    H = np.zeros((m, m))
    start_idx = 1

    if start_idx == m:
        print("Warning: restart array is full, no iterations performed.")

    V[:, 0] = v0
    for j in range(start_idx, m + 1):
        # Apply iteration M @ f = A @ v by solving the linear system
        if inverse:
            # Solve the system A @ f = M @ v: equivalent to f = A^{-1} @ M @ v
            # Will converge towards the smallest eigenvalues of A
            f = sparse.linalg.spsolve(A, M @ V[:, j - 1])
        else:
            # Solve the system M @ f = A @ v: equivalent to f = M^{-1} @ A @ v
            # Will converge towards the largest eigenvalues of A
            f = sparse.linalg.spsolve(M, A @ V[:, j - 1])

        # Gram-Schmidt
        h = np.zeros(j)
        for i in range(j):
            h[i] = np.dot(V[:, i], f)
            f = f - V[:, i] * h[i]

        # Fill in the upper Hessenberg matrix
        H[:j, j - 1] = h

        # Norm of the orthogonal residual
        beta = np.linalg.norm(f)

        if j < m:
            H[j, j - 1] = beta

        # Normalize the orthogonal residual for use as a new
        # Krylov basis vector
        V[:, j] = f / beta

    return V, H, beta


def eig_arnoldi(A, M=None, v0=None, m=100, sigma=0.0):
    n = A.shape[0]
    if v0 is None:
        v0 = np.random.rand(n)
    v0 = v0 / np.linalg.norm(v0)
    if M is None:
        M = sparse.eye(n)
    A = sparse.csc_matrix(A - sigma * M)  # Spectral shift
    M = sparse.csc_matrix(M)
    V, H, _ = arnoldi(A, M, v0, m)
    ritz_vals, ritz_vecs = linalg.eig(H)
    evecs = V[:, :m] @ ritz_vecs
    evals = 1 / (ritz_vals + sigma)  # Undo spectral shift
    return evals, evecs


if __name__ == "__main__":
    # 1. Set up the finite difference matrix for the heat equation Dxx
    n = 200
    L = 1.0
    A = sparse_fd_matrix(n, L / n)
    M = sparse.eye(n - 1)  # Mass matrix

    # 3. Eigenvalues of the finite difference matrix (dense eig)
    evals, evecs = linalg.eigh(A.toarray())
    sort_idx = np.argsort(-evals)
    dense_evals = evals[sort_idx]
    dense_evecs = evecs[:, sort_idx]

    # 4. Eigenvalues of the finite difference matrix (Arnoldi)
    arn_evals, arn_evecs = eig_arnoldi(A)
    sort_idx = np.argsort(-abs(arn_evals))
    arn_evals_dt = arn_evals[sort_idx]

    # # 5. Eigenvalues of the finite difference matrix (Krylov-Schur)
    # ks_evals_dt, ks_evecs = eig_ks(M, delta=0.99, m=10, n_evals=5)
    # sort_idx = np.argsort(-abs(ks_evals_dt))
    # ks_evals_dt = ks_evals_dt[sort_idx]
    # ks_evecs = ks_evecs[:, sort_idx]
    # ks_evals = np.log(ks_evals_dt) / dt

    # Print the leading eigenvalues for each method
    n_print = 5
    print(f"Dense eigenvalues: {dense_evals[:n_print]}")
    arn_err = abs(dense_evals[:n_print] - arn_evals[:n_print].real)
    print(f"Arnoldi eigenvalues: {arn_evals[:n_print].real}")
    print(f"    error: {arn_err}")
    # ks_err = abs(dense_evals[:n_print] - ks_evals[:n_print].real)
    # print(f"Krylov-Schur eigenvalues: {ks_evals[:n_print].real}")
    # print(f"    error: {ks_err}")
