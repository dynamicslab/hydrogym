"""Eigendecomposition with Krylov-Schur iteration"""

import firedrake as fd
import numpy as np
from cyl_common import (
    FlowPropagator,
    base_checkpoint,
    evec_checkpoint,
    flow,
    inner_product,
)
from eig import eig_ks

if __name__ == "__main__":
    flow.load_checkpoint(base_checkpoint)
    qB = flow.q.copy(deepcopy=True)

    tau = 0.05
    dt = 0.01
    A = FlowPropagator(flow, qB, dt, tau)

    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(flow.mixed_space)
    v0.assign(v0 / inner_product(v0, v0) ** 0.5)
    n_evals = 6
    n_save = min(n_evals, 20)
    rvals, evecs_real, evecs_imag = eig_ks(
        A,
        v0,
        m=64,
        inner_product=inner_product,
        debug_tau=tau,
        n_evals=n_evals,
        tol=1e-6,
        delta=0.05,
    )
    ks_evals = np.log(rvals) / tau

    print(f"Krylov-Schur eigenvalues: {ks_evals[:n_save]}")

    # Save checkpoints
    with fd.CheckpointFile(evec_checkpoint, "w") as chk:
        for i in range(n_save):
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
