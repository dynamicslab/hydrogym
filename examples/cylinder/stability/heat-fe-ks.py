import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import tripcolor, triplot
from scipy import linalg, sparse
from ufl import cos, div, dot, dx, exp, grad, inner

import hydrogym.firedrake as hgym


class HeatPropagator:
    def __init__(self, fn_space, dt, tau, k):
        u = fd.TrialFunction(fn_space)
        v = fd.TestFunction(fn_space)
        self.n_steps = int(tau // dt)
        self.u_prev = fd.Function(fn_space)
        self.a = inner(u, v) * dx + dt * k * inner(grad(u), grad(v)) * dx
        self.L = inner(self.u_prev, v) * dx
        self.bcs = [fd.DirichletBC(fn_space, 0, "on_boundary")]
        self.solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        self.fn_space = fn_space

    def __matmul__(self, u):
        self.u_prev.assign(u)
        u_sol = u.copy(deepcopy=True)
        for _ in range(self.n_steps):
            fd.solve(
                self.a == self.L,
                u_sol,
                bcs=self.bcs,
                solver_parameters=self.solver_parameters,
            )
            self.u_prev.assign(u_sol)
        return u_sol


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

        if start_idx == m:
            print("Warning: restart array is full, no iterations performed.")

        f = self.copy(v0)
        v = self.copy(v0)
        for j in range(start_idx, m):
            beta = self.norm(f)
            H[j, j - 1] = beta
            self.assign(v, f / beta)
            self.assign(V[j], v)
            w = A @ v
            self.assign(f, w)
            for k in range(j + 1):
                H[k, j] = self.inner(V[k], w)
                self.assign(f, f - H[k, j] * V[k])

        return V, H, v, beta


class FiredrakeArnoldi(ArnoldiBase):
    def inner(self, u, v):
        return fd.assemble(inner(u, v) * dx)

    def assign(self, u, v):
        u.assign(v)

    def copy(self, u):
        return u.copy(deepcopy=True)


def eig_arnoldi(A, v0, m=100, sort=None):
    if sort is None:
        # Decreasing magnitude
        def sort(x):
            return np.argsort(-abs(x))

    arnoldi = FiredrakeArnoldi()
    V, H, _, _ = arnoldi(A, v0, m)
    ritz_vals, ritz_vecs = linalg.eig(H)

    sort_idx = sort(ritz_vals)
    ritz_vals = ritz_vals[sort_idx]
    ritz_vecs = ritz_vecs[:, sort_idx]

    fn_space = v0.function_space()
    evecs_real = [fd.Function(fn_space) for _ in range(len(V))]
    evecs_imag = [fd.Function(fn_space) for _ in range(len(V))]
    for i in range(len(V)):
        evecs_real[i].assign(sum(ritz_vecs[j, i].real * V[j] for j in range(len(V))))
        evecs_imag[i].assign(sum(ritz_vecs[j, i].imag * V[j] for j in range(len(V))))
    return ritz_vals, evecs_real, evecs_imag


if __name__ == "__main__":
    tau = 0.1
    dt = 0.01
    k = 0.1
    mesh = fd.UnitSquareMesh(10, 10)
    fn_space = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    A = HeatPropagator(fn_space, dt, tau, k)

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(fn_space)
    rvals, evecs_real, evecs_imag = eig_arnoldi(A, v0, m=20)
    evals = np.log(rvals) / dt
    print(evals[:10])

    # Plot the eigenmodes
    n_evals_plt = 5
    fig, axs = plt.subplots(1, n_evals_plt, figsize=(12, 2), sharex=True, sharey=True)

    for i in range(n_evals_plt):
        tripcolor(evecs_real[i], axes=axs[i], cmap="RdBu")
    plt.show()
