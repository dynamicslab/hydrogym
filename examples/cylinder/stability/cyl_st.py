import firedrake as fd
import numpy as np
from cyl_common import base_checkpoint, evec_checkpoint, flow, inner_product
from ufl import dx, inner

import hydrogym.firedrake as hgym

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

    (uB, pB) = fd.split(qB)
    q_trial = fd.TrialFunction(fn_space)
    q_test = fd.TestFunction(fn_space)
    (v, s) = fd.split(q_test)

    # Linear form expressing the _LHS_ of the Navier-Stokes without time derivative
    # For a steady solution this is F(qB) = 0.
    # TODO: Make this a standalone function - could be used in NewtonSolver and transient
    newton_solver = hgym.NewtonSolver(flow)
    F = newton_solver.steady_form((uB, pB), q_test=(v, s))
    # The Jacobian of F is the bilinear form J(qB, q_test) = dF/dq(qB) @ q_test
    J = -fd.derivative(F, qB, q_trial)

    # Uncomment to solve the adjoint problem (this is in utils.linalg)
    # J = hgym.utils.linalg.adjoint(J)

    def M(q):
        u = q.subfunctions[0]
        return inner(u, v) * dx

    def solve_inverse(v1, v0):
        """Solve the matrix pencil A @ v1 = M @ v0 for v1.

        This is equivalent to the "inverse iteration" v1 = (A^{-1} @ M) @ v0

        Stores the result in `v1`
        """
        fd.solve(J == M(v0), v1, bcs=bcs, solver_parameters=solver_parameters)

    # Need to construct
    rng = fd.RandomGenerator(fd.PCG64())
    v0 = rng.standard_normal(fn_space)
    alpha = np.sqrt(inner_product(v0, v0))
    v0.assign(v0 / alpha)

    evals, evecs_real, evecs_imag = hgym.utils.eig_arnoldi(
        solve_inverse, v0, inner_product, m=100
    )

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
