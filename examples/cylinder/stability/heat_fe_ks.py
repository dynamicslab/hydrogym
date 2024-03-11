import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from eig import eig_arnoldi, eig_ks
from firedrake.pyplot import tripcolor, triplot
from ufl import dx, grad, inner


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
    arn_rvals, arn_evecs_real, arn_evecs_imag = eig_arnoldi(A, v0, m=20)
    arn_evals = np.log(arn_rvals) / dt

    ks_rvals, ks_evecs_real, ks_evecs_imag = eig_ks(
        A, v0, m=20, delta=0.9, tol=1e-10, n_evals=10
    )
    ks_evals = np.log(ks_rvals) / dt

    n_print = 5
    print(f"Arnoldi eigenvalues: {arn_evals[:n_print].real}")
    print(f"Krylov-Schur eigenvalues: {ks_evals[:n_print].real}")

    # Plot the eigenmodes
    n_evals_plt = 5
    fig, axs = plt.subplots(1, n_evals_plt, figsize=(12, 2), sharex=True, sharey=True)

    for i in range(n_evals_plt):
        tripcolor(ks_evecs_real[i], axes=axs[i], cmap="RdBu")
    plt.show()
