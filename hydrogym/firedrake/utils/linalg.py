from __future__ import annotations

import abc
import dataclasses

import firedrake as fd
import numpy as np
import ufl

from firedrake.petsc import PETSc
from slepc4py import SLEPc

__all__ = ["eig"]

MUMPS_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

DEFAULT_EPS_PARAMETERS = {
    "eps_gen_non_hermitian": None,
    "eps_target": "0",
    "eps_type": "krylovschur",
    "eps_largest_real": True,
    "st_type": "sinvert",
    "st_pc_factor_shift_type": "NONZERO",
}


class LinearOperator(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __matmul__(self, v: fd.Function):
    """Return the matrix-vector product A @ v."""
    pass


@dataclasses.dataclass
class DirectOperator(LinearOperator):
  J: ufl.Form
  M: ufl.Form
  bcs: list[fd.DirichletBC]
  function_space: fd.FunctionSpace
  solver_parameters: dict = None
  copy_output: bool = False

  def __post_init__(self):
    self._v0 = fd.Function(self.function_space)
    self._v1 = fd.Function(self.function_space)

    q_trial = fd.TrialFunction(self.function_space)
    q_test = fd.TestFunction(self.function_space)

    L = ufl.action(self.J, self._v0)  # linear form
    a = ufl.inner(q_trial, q_test) * ufl.dx  # bilinear form
    self._problem = fd.LinearVariationalProblem(
        a, L, self._v1, bcs=self.bcs, constant_jacobian=True)

    if self.solver_parameters is None:
      self.solver_parameters = MUMPS_SOLVER_PARAMETERS

    self._solver = fd.LinearVariationalSolver(
        self._problem, solver_parameters=self.solver_parameters)

  def __matmul__(self, v: fd.Function):
    self._v0.assign(v)

    self._solver.solve()
    if self.copy_output:
      return self._v1.copy(deepcopy=True)
    return self._v1

  @property
  def T(self) -> DirectOperator:
    """Return the adjoint operator."""
    cls = self.__class__
    args = self.J.arguments()
    JT = ufl.adjoint(self.J, reordered_arguments=(args[0], args[1]))
    return cls(JT, self.M, self.bcs, self.function_space)


def eig(lin_op: DirectOperator,
        n: int = 1,
        sigma: complex = None,
        tol: float = 1e-10,
        krylov_dim: int = 100,
        slepc_options: dict = None) -> tuple[np.ndarray, list[fd.Function]]:
  """Eigenvalue decomposition of the matrix pencil `(A, M)` using SLEPc.

  For stability analysis, typically `A` is the dynamics matrix and `M` is the mass
  matrix.  This function dispatches to SLEPc for the actual iterative eigenvalue
  solver

  The default behavior is to use a shift-invert transformation to avoid inverting `M`,
  which is singular in the case of the incompressible Navier-Stokes equations.
  Ultimately this computes `[A - sigma * M]^-1` instead of `M^-1*A`, so that the
  eigenvalues which are fastest to converge are those nearest to `sigma`.

  Note that the standard firedrake (and PETSc) build does not support complex
  numbers, so that in order to use complex-valued shifts, the complex PETSc and
  firedrake builds should be used.

  Args:
      A: The dynamics matrix, specified as a UFL form.
      M: The mass matrix, specified as a UFL form.
      bcs: The Dirichlet boundary conditions to apply to the matrices.
      n: The number of eigenvalues to converge.
      krylov_dim: The dimension of the Krylov subspace (number of Arnoldi vectors).
      sigma: The shift for the shift-invert Arnoldi method.
      tol: Tolerance to use for determining converged eigenvalues.
      slepc_options: Additional options to pass to SLEPc.

  Returns:
      evals: The converged eigenvalues.
      evecs: The eigenvectors corresponding to the converged eigenvalues.
  """

  # fn_space = lin_op.function_space

  A = fd.assemble(lin_op.J, bcs=lin_op.bcs)
  M = fd.assemble(lin_op.M, bcs=lin_op.bcs)

  if sigma is not None and sigma.imag != 0:
    if PETSc.ScalarType is not np.complex128:
      raise ValueError(
          "Complex PETSc build required for complex shift-invert")

  if slepc_options is None:
    slepc_options = DEFAULT_EPS_PARAMETERS

  slepc_options = {
      **slepc_options,
      "eps_target": f"{sigma.real}+{sigma.imag}i",
      "eps_tol": tol,
  }

  # SLEPc Setup
  opts = PETSc.Options()
  for key, val in slepc_options.items():
    opts.setValue(key, val)

  es = SLEPc.EPS().create(comm=fd.COMM_WORLD)
  es.setDimensions(n, krylov_dim)
  es.setOperators(A.petscmat, M.petscmat)
  es.setFromOptions()
  es.solve()

  vr, vi = A.petscmat.getVecs()  # Get vectors compatible with the matrix

  n_conv = es.getConverged()
  evals = np.array([es.getEigenvalue(i) for i in range(n_conv)])
  evecs = [fd.Function(lin_op.function_space) for _ in range(n_conv)]
  for i in range(n_conv):
    es.getEigenpair(i, vr, vi)
    with evecs[i].dat.vec as vec:
      vec.setArray(vr.array)
  return evals, evecs
