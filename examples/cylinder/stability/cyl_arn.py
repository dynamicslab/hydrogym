"""Eigendecomposition with Arnoldi iteration"""

import firedrake as fd
import numpy as np
from cyl_common import (
    FlowPropagator,
    base_checkpoint,
    evec_checkpoint,
    flow,
    inner_product,
)
from eig import eig_arnoldi

if __name__ == "__main__":
    flow.load_checkpoint(base_checkpoint)
    qB = flow.q.copy(deepcopy=True)

    tau = 0.5
    dt = 0.01
    A = FlowPropagator(flow, qB, dt, tau)

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(flow.mixed_space)
    v0.assign(v0 / inner_product(v0, v0) ** 0.5)
    rvals, evecs_real, evecs_imag = eig_arnoldi(
        A,
        v0,
        m=256,
        inner_product=inner_product,
        debug_tau=tau,
    )
    arn_evals = np.log(rvals) / tau

    n_print = 64
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
