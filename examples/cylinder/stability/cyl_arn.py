"""Eigendecomposition with Arnoldi iteration"""

import firedrake as fd
import numpy as np
from cyl_common import base_checkpoint, evec_checkpoint, flow, stabilization
from eig import eig_arnoldi

# from scipy import linalg
from ufl import dx, inner

import hydrogym.firedrake as hgym


class FlowPropagator:
    def __init__(self, flow, qB, dt, tau):
        self.flow = flow
        self.tau = tau
        self.solver = hgym.LinearizedBDF(
            flow,
            dt,
            qB=qB,
            stabilization=stabilization,
        )

    def __matmul__(self, q):
        self.flow.q.assign(q)

        # Assign the current solution to all `u_prev`
        # TODO: Move to solver reset function?
        for u in self.solver.u_prev:
            u.assign(q.subfunctions[0])

        self.solver.solve((0.0, self.tau))
        return self.flow.q.copy(deepcopy=True)  # TODO: do we need deepcopy?


def inner_product(q1, q2):
    u1 = q1.subfunctions[0]
    u2 = q2.subfunctions[0]
    return fd.assemble(inner(u1, u2) * dx)


if __name__ == "__main__":
    flow.load_checkpoint(base_checkpoint)
    qB = flow.q.copy(deepcopy=True)

    tau = 0.05
    dt = 0.0025
    A = FlowPropagator(flow, qB, dt, tau)

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(flow.mixed_space)
    rvals, evecs_real, evecs_imag = eig_arnoldi(
        A,
        v0,
        m=256,
        inner_product=inner_product,
        debug_tau=tau,
    )
    arn_evals = np.log(rvals) / tau

    n_print = 10
    print(f"Arnoldi eigenvalues: {arn_evals[:n_print]}")

    # Save checkpoints
    with fd.CheckpointFile(evec_checkpoint, "w") as chk:
        for i in range(n_print):
            evecs_real[i].rename(f"evec_{i}_re")
            chk.save_function(evecs_real[i])
            evecs_imag[i].rename(f"evec_{i}_im")
            chk.save_function(evecs_imag[i])

    # print(f"Krylov-Schur eigenvalues: {ks_evals[:n_print].real}")

    # # Plot the eigenmodes
    # n_evals_plt = 5
    # fig, axs = plt.subplots(1, n_evals_plt, figsize=(12, 2), sharex=True, sharey=True)

    # for i in range(n_evals_plt):
    #     tripcolor(ks_evecs_real[i], axes=axs[i], cmap="RdBu")
    # plt.show()
