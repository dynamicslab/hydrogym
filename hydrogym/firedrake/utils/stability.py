"""Utilities for global stability analysis

This file specializes some of the Arnoldi iteration in `eig.py`
for incompressible Navier-Stokes equations.
"""
import firedrake as fd
import ufl

from .eig import ArnoldiIterator

MUMPS_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def _real_inner_product(q1, q2, assemble=True):
    """Energy inner product for the Navier-Stokes equations"""
    (u, _) = fd.split(q1)
    (v, _) = fd.split(q2)
    M = ufl.inner(u, v) * ufl.dx
    if not assemble:
        return M
    return fd.assemble(M)


def _make_real_inv_iterator(flow, sigma, adjoint=False, solver_parameters=None):
    """Construct a shift-inverse Arnoldi iterator with real (or zero) shift.

    The shift-inverse iteration solves the matrix pencil
    (J - sigma * M) @ v1 = M @ v0 for v1, where J is the Jacobian of the
    Navier-Stokes equations, and M is the mass matrix.
    n"""
    if solver_parameters is None:
        solver_parameters = MUMPS_SOLVER_PARAMETERS

    # Set up function spaces
    fn_space = flow.mixed_space
    qB = flow.q
    (uB, pB) = fd.split(qB)
    q_trial = fd.TrialFunction(fn_space)
    q_test = fd.TestFunction(fn_space)
    (v, s) = fd.split(q_test)

    # Collect boundary conditions
    flow.linearize_bcs()
    bcs = flow.collect_bcs()

    def M(q):
        """Mass matrix for the Navier-Stokes equations"""
        return _real_inner_product(q, q_test, assemble=False)

    # Linear form expressing the RHS of the Navier-Stokes without time derivative
    # For a steady solution this is F(qB) = 0.
    F = flow.residual((uB, pB), q_test=(v, s))

    if sigma != 0.0:
        F -= sigma * M(qB)

    # The Jacobian of F is the bilinear form J(qB, q_test) = dF/dq(qB) @ q_test
    J = fd.derivative(F, qB, q_trial)

    if adjoint:
        args = J.arguments()
        J = ufl.adjoint(J, reordered_arguments=(args[0], args[1]))

    def _solve_inverse(v1, v0):
        """Solve the matrix pencil A @ v1 = M @ v0 for v1.

        This is equivalent to the "inverse iteration" v1 = (A^{-1} @ M) @ v0

        Stores the result in `v1`
        """
        fd.solve(J == M(v0), v1, bcs=bcs, solver_parameters=solver_parameters)

    def _random_vec(rng_seed=None):
        rng = fd.RandomGenerator(fd.PCG64(seed=rng_seed))
        v0 = rng.standard_normal(fn_space)
        alpha = fd.sqrt(_real_inner_product(v0, v0))
        v0.assign(v0 / alpha)
        return v0

    return ArnoldiIterator(
        _solve_inverse,
        _real_inner_product,
        _random_vec,
        sort="lr",
        inverse=True,
    )


def _complex_inner_product(q1, q2, assemble=True):
    u1_re, _p1_re, u1_im, _p1_im = fd.split(q1)
    u2_re, _p2_re, u2_im, _p2_im = fd.split(q2)
    M = (ufl.inner(u1_re, u2_re) + ufl.inner(u1_im, u2_im)) * ufl.dx
    if not assemble:
        return M
    return fd.assemble(M)


def _make_complex_inv_iterator(flow, sigma, adjoint=False, solver_parameters=None):
    """Construct a shift-inverse Arnoldi iterator with complex-valued shift.

    The shifted bilinear form is `A = (J - sigma * M)`
    For sigma = (sr, si), the real and imaginary parts of A are
    A = (J - sr * M, -si * M)
    The system solve is A @ v1 = M @ v0, so for complex vectors v = (vr, vi):
    (Ar + 1j * Ai) @ (v1r + 1j * v1i) = M @ (v0r + 1j * v0i)
    Since we can't do complex analysis without re-building PETSc, instead we treat this
    as a block system:
    ```
      [Ar,   Ai]  [v1r]  = [M 0]  [v0r]
      [-Ai,  Ar]  [v1i]    [0 M]  [v0i]
    ```

    Note that this will be more expensive than real- or zero-shifted iteration,
    since there are twice as many degrees of freedom.  However, it will tend to
    converge faster for the eigenvalues of interest.

    The shift-inverse iteration solves the matrix pencil
    (J - sigma * M) @ v1 = M @ v0 for v1, where J is the Jacobian of the
    Navier-Stokes equations, and M is the mass matrix.
    n"""
    if solver_parameters is None:
        solver_parameters = MUMPS_SOLVER_PARAMETERS

    # Set up function spaces
    fn_space = flow.mixed_space
    W = fn_space * fn_space
    V1, Q1, V2, Q2 = W

    # Set the boundary conditions for each function space
    # These will be identical
    flow.linearize_bcs(function_spaces=(V1, Q1))
    bcs1 = flow.collect_bcs()

    flow.linearize_bcs(function_spaces=(V2, Q2))
    bcs2 = flow.collect_bcs()

    bcs = [*bcs1, *bcs2]

    # Since the base flow is used to construct the Navier-Stokes Jacobian
    # which is used on the diagonal block for both real and imaginary components,
    # we have to duplicate the base flow for both components.  This does NOT
    # mean that the base flow literally has an imaginary component
    qB = fd.Function(W)
    uB_re, pB_re, uB_im, pB_im = fd.split(qB)
    qB.subfunctions[0].interpolate(flow.q.subfunctions[0])
    qB.subfunctions[1].interpolate(flow.q.subfunctions[1])
    qB.subfunctions[2].interpolate(flow.q.subfunctions[0])
    qB.subfunctions[3].interpolate(flow.q.subfunctions[1])

    # Create trial and test functions
    q_trial = fd.TrialFunction(W)
    q_test = fd.TestFunction(W)
    (u_re, p_re, u_im, p_im) = fd.split(q_trial)
    (v_re, s_re, v_im, s_im) = fd.split(q_test)

    # Construct the nonlinear residual for the Navier-Stokes equations
    F_re = flow.residual((uB_re, pB_re), q_test=(v_re, s_re))
    F_im = flow.residual((uB_im, pB_im), q_test=(v_im, s_im))

    def _inner(u, v):
        return ufl.inner(u, v) * ufl.dx

    # Shift each block of the linear form appropriately
    F11 = F_re - sigma.real * _inner(uB_re, v_re)
    F22 = F_im - sigma.real * _inner(uB_im, v_im)
    F12 = sigma.imag * _inner(uB_im, v_re)
    F21 = -sigma.imag * _inner(uB_re, v_im)
    F = F11 + F22 + F12 + F21

    # Differentiate to get the bilinear form for the Jacobian
    J = fd.derivative(F, qB, q_trial)

    if adjoint:
        args = J.arguments()
        J = ufl.adjoint(J, reordered_arguments=(args[0], args[1]))

    def M(q):
        """Mass matrix for the Navier-Stokes equations"""
        return _complex_inner_product(q, q_test, assemble=False)

    def _solve_inverse(v1, v0):
        """Solve the matrix pencil A @ v1 = M @ v0 for v1.

        This is equivalent to the "inverse iteration" v1 = (A^{-1} @ M) @ v0

        Stores the result in `v1`
        """
        fd.solve(J == M(v0), v1, bcs=bcs, solver_parameters=solver_parameters)

    def _random_vec(rng_seed=None):
        rng = fd.RandomGenerator(fd.PCG64(seed=rng_seed))
        v0 = rng.standard_normal(W)
        alpha = fd.sqrt(_complex_inner_product(v0, v0))
        v0.assign(v0 / alpha)
        return v0

    return ArnoldiIterator(
        _solve_inverse,
        _complex_inner_product,
        _random_vec,
        sort="lr",
        inverse=True,
    )


def make_st_iterator(flow, sigma=0.0, adjoint=False, solver_parameters=None):
    """Construct a spectrally-transformed (shift-invert) Arnoldi iterator"""
    if sigma.imag == 0.0:
        return _make_real_inv_iterator(
            flow, sigma.real, adjoint=adjoint, solver_parameters=solver_parameters
        )
    return _make_complex_inv_iterator(
        flow, sigma, adjoint=adjoint, solver_parameters=solver_parameters
    )
