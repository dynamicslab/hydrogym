import firedrake as fd
import numpy as np
from firedrake import dx, inner
from scipy import linalg

from hydrogym.firedrake import print as parprint


class ArnoldiBase:
    def inner(self, u, v):
        return np.dot(u, v)

    def norm(self, u):
        return np.sqrt(self.inner(u, u))

    def assign(self, u, v):
        u[:] = v

    def copy(self, u):
        return np.array(u)

    def __call__(self, A, v0, m=100, restart=None, debug_tau=None):
        if restart is None:
            # Initial value here doesn't matter - all data will be overwritten
            V = [self.copy(v0).assign(0.0) for _ in range(m + 1)]
            self.assign(V[0], v0)
            H = np.zeros((m, m))
            start_idx = 1
        else:
            V, H, start_idx = restart
            print(f"Restarting from previous iteration with p={start_idx}")

        if start_idx == m:
            print("Warning: restart array is full, no iterations performed.")

        for j in range(start_idx, m + 1):
            f = A @ V[j - 1]

            # Gram-Schmidt
            h = np.zeros(j)
            for i in range(j):
                h[i] = self.inner(V[i], f)
                self.assign(f, f - V[i] * h[i])

            # Fill in the upper Hessenberg matrix
            H[:j, j - 1] = h

            # Norm of the orthogonal residual
            beta = self.norm(f)

            if j < m:
                H[j, j - 1] = beta

            # Normalize the orthogonal residual for use as a new
            # Krylov basis vector
            self.assign(V[j], f / beta)

            # DEBUG: Print the eigenvalues
            if debug_tau is not None:
                parprint(f"*** Arnoldi iteration {j} ***")
                ritz_vals, ritz_vecs = linalg.eig(H[:j, :j])
                sort_idx = np.argsort(-abs(ritz_vals))
                ritz_vals = ritz_vals[sort_idx]
                ritz_vecs = ritz_vecs[:, sort_idx]
                parprint(f"Eigvals: {np.log(ritz_vals[:20]) / debug_tau}")
                res = abs(beta * ritz_vecs[-1, :20])
                parprint(f"Residuals: {res}")

        return V, H, beta


class FiredrakeArnoldi(ArnoldiBase):
    def __init__(self, inner_product=None):
        if inner_product is None:

            def _inner(u, v):
                return fd.assemble(inner(u, v) * dx)

        else:
            _inner = inner_product
        self._inner = _inner

    def inner(self, u, v):
        return self._inner(u, v)

    def assign(self, u, v):
        u.assign(v)

    def copy(self, u):
        return u.copy(deepcopy=True)


def eig_arnoldi(A, v0, m=100, sort=None, inner_product=None, debug_tau=None):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-abs(x))

    arnoldi = FiredrakeArnoldi(inner_product=inner_product)
    V, H, _beta = arnoldi(A, v0, m, debug_tau=debug_tau)
    ritz_vals, ritz_vecs = linalg.eig(H)

    sort_idx = sort(ritz_vals)
    ritz_vals = ritz_vals[sort_idx]
    ritz_vecs = ritz_vecs[:, sort_idx]

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(m)]
    evecs_imag = [fd.Function(fn_space) for _ in range(m)]
    for i in range(m):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(m)))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(m)))
    return ritz_vals, evecs_real, evecs_imag


def eig_ks(
    A,
    v0,
    n_evals=10,
    m=100,
    sort=None,
    tol=1e-10,
    delta=0.1,
    maxiter=None,
    inner_product=None,
    debug_tau=None,
):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return abs(x) > 1.0 - delta

    arnoldi = FiredrakeArnoldi(inner_product=inner_product)
    fn_space = v0.function_space()

    restart = None
    is_converged = False
    k = 0  # Iteration count (for logging)
    while not is_converged:
        V, H, beta = arnoldi(A, v0, m, restart=restart, debug_tau=debug_tau)

        # Check for convergence based on Arnoldi residuals
        ritz_vals, ritz_vecs = linalg.eig(H)

        converged = []
        for i in range(m):
            y = ritz_vecs[:, i]
            # print(f"   Ritz vector {i}: {y}")
            mu = ritz_vals[i]
            res = abs(beta * y[-1])
            if res < tol:
                converged.append((mu, y))

        print(f"Iteration {k}: {len(converged)} converged")

        # If enough eigenvalues have converged, return them and exit
        if len(converged) >= n_evals or (maxiter is not None and k >= maxiter):
            is_converged = True
            print(f"Converged: {len(converged)} eigenvalues")

            p = len(converged)
            evals = np.zeros(p, dtype=np.complex128)
            evecs_real = [fd.Function(fn_space) for _ in range(p)]
            evecs_imag = [fd.Function(fn_space) for _ in range(p)]
            for i, (mu, y) in enumerate(converged):
                evals[i] = mu
                evecs_real[i].assign(sum(V[j] * y[j].real for j in range(m)))
                evecs_imag[i].assign(sum(V[j] * y[j].imag for j in range(m)))
            return evals, evecs_real, evecs_imag

        # Schur decomposition
        S, Q, p = linalg.schur(H, sort=sort)
        print(f"   Schur form: {p} retained eigenvalues")

        # Keep the "wanted" part of the Schur form. These will be the eigenvalues
        # that are closest to the unit circle (least stable).
        S[p:, :p] = 0
        S[:p, p:] = 0
        S[p:, p:] = 0

        # Re-order the Krylov basis.  Since Q is real, these fields are also real.
        Vp = [fd.Function(fn_space) for _ in range(m + 1)]
        v0.assign(sum(V[j] * Q[j, p] for j in range(m)))
        for i in range(p):
            Vp[i].assign(sum(V[j] * Q[j, i] for j in range(m)))

        # Update the matrix with the "b" vector for residuals
        b = np.zeros(m)
        b[-1] = beta
        b = Q.T @ b
        S[p, :p] = b[:p]
        Vp[p].assign(V[-1])  # Restart from the last Krylov basis vector

        # p += 1 ??

        # Restart with the wanted part of the Krylov basis
        restart = (Vp, S, p)
        k += 1
