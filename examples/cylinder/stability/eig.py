import firedrake as fd
import numpy as np
from firedrake import dx, inner
from scipy import linalg


class ArnoldiBase:
    def inner(self, u, v):
        return np.dot(u, v)

    def norm(self, u):
        return np.sqrt(self.inner(u, u))

    def assign(self, u, v):
        u[:] = v

    def copy(self, u):
        return np.array(u)

    def __call__(self, A, v0, m=100, restart=None):
        if restart is None:
            # Initial value here doesn't matter - all data will be overwritten
            V = [self.copy(v0) for _ in range(m)]
            H = np.zeros((m, m))
            start_idx = 0
        else:
            V, H = restart
            start_idx = len(V)
            V.extend([self.copy(v0) for _ in range(m - start_idx)])
            H = np.pad(H, ((0, m - start_idx), (0, m - start_idx)), mode="constant")
            print(f"Restarting from previous iteration with p={start_idx}")

        if start_idx == m:
            print("Warning: restart array is full, no iterations performed.")

        v = self.copy(v0)
        f = self.copy(v0)
        self.assign(v, v / self.norm(v0))
        w = A @ v
        alpha = self.inner(v, w)
        self.assign(f, w - v * alpha)
        self.assign(V[start_idx], v)
        H[start_idx, start_idx] = alpha
        for j in range(start_idx + 1, m):
            print(f"Arnoldi iteration {j}")
            beta = self.norm(f)
            H[j, j - 1] = beta
            self.assign(v, f / beta)
            self.assign(V[j], v)
            w = A @ v
            self.assign(f, w)
            for k in range(j + 1):
                H[k, j] = self.inner(V[k], w)
                self.assign(f, f - V[k] * H[k, j])

            # DEBUG: Print the eigenvalues
            tau = 0.1
            ritz_vals, ritz_vecs = linalg.eig(H[: j + 1, : j + 1])
            sort_idx = np.argsort(-abs(ritz_vals))
            ritz_vals = ritz_vals[sort_idx]
            print(np.log(ritz_vals[:10]) / tau)

        return V, H, v, beta


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


def eig_arnoldi(A, v0, m=100, sort=None, inner_product=None):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-abs(x))

    arnoldi = FiredrakeArnoldi(inner_product=inner_product)
    V, H, _, _ = arnoldi(A, v0, m)
    ritz_vals, ritz_vecs = linalg.eig(H)

    sort_idx = sort(ritz_vals)
    ritz_vals = ritz_vals[sort_idx]
    ritz_vecs = ritz_vecs[:, sort_idx]

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(len(V))]
    evecs_imag = [fd.Function(fn_space) for _ in range(len(V))]
    for i in range(len(V)):
        evecs_real[i].assign(sum(V[j] * ritz_vecs[j, i].real for j in range(len(V))))
        evecs_imag[i].assign(sum(V[j] * ritz_vecs[j, i].imag for j in range(len(V))))
    return ritz_vals, evecs_real, evecs_imag


def eig_ks(
    A, v0, n_evals=10, m=100, tol=1e-10, delta=0.05, maxiter=None, inner_product=None
):
    arnoldi = FiredrakeArnoldi(inner_product=inner_product)
    fn_space = v0.function_space()

    restart = None
    k = 0  # Iteration count (for logging)

    converged = []
    while len(converged) < n_evals:
        V, H, v0, beta = arnoldi(A, v0, m, restart=restart)

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
            # if True:
            if maxiter is not None and k >= maxiter:
                print(f"Maximum number of iterations reached ({maxiter}). Exiting.")
            p = len(converged)
            evals = np.zeros(p, dtype=np.complex128)
            evecs_real = [fd.Function(fn_space) for _ in range(p)]
            evecs_imag = [fd.Function(fn_space) for _ in range(p)]
            for i, (mu, y) in enumerate(converged):
                evals[i] = mu
                evecs_real[i].assign(sum(y[j].real * V[j] for j in range(m)))
                evecs_imag[i].assign(sum(y[j].imag * V[j] for j in range(m)))
            return evals, evecs_real, evecs_imag

        # Schur decomposition
        S, Q, p = linalg.schur(H, sort=lambda x: abs(x) > 1.0 - delta)

        # Keep the "wanted" part of the Schur form. These will be the eigenvalues
        # that are closest to the unit circle (least stable).
        Hp = S[:p, :p]

        # Re-order the Krylov basis.  Since Q is real, these fields are also real.
        Vp = [fd.Function(fn_space) for _ in range(p)]
        for i in range(p):
            Vp[i].assign(sum(Q[j, i] * V[j] for j in range(m)))

        # Restart with the wanted part of the Krylov basis
        restart = (Vp, Hp)
        k += 1
