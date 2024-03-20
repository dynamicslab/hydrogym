from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Callable, Iterable

import firedrake as fd
import numpy as np
import ufl
from firedrake import logging
from modred import PODHandles, VectorSpaceHandles
from scipy import sparse

from .modred_interface import Snapshot, vec_handle_mean

if TYPE_CHECKING:
  from hydrogym.firedrake import FlowConfig

MUMPS_SOLVER_PARAMETERS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


class LinearOperator(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __matmul__(self, v: fd.Function):
    """Return the matrix-vector product A @ v."""
    pass


class DirectOperator(LinearOperator):
  J: ufl.Form
  bcs: list[fd.DirichletBC]
  function_space: fd.FunctionSpace

  def __matmul__(self, v: fd.Function):
    f_bar = fd.assemble(self.J @ v, bcs=self.bcs)  # Cofunction
    return fd.Function(self.function_space, val=f_bar.dat)

  @property
  def T(self) -> InverseOperator:
    """Return the adjoint operator.

        This will solve the matrix pencil A^T @ f = M^T @ v0 for f.
        """
    cls = self.__class__
    args = self.J.arguments()
    JT = ufl.adjoint(self.J, reordered_arguments=(args[0], args[1]))
    return cls(JT, self.bcs, self.function_space)


@dataclasses.dataclass
class InverseOperator(LinearOperator):
  """A simple wrapper for the inverse of a matrix pencil.

    Note that this object will own the output Function unless
    `copy_output=True` is set.  This is memory-efficient, but could
    lead to confusion if the output reference is modified. The Arnoldi
    iteration algorithm is written so this isn't a problem.
    """

  J: ufl.Form
  M: Callable[[fd.Function], ufl.Form]
  bcs: list[fd.DirichletBC]
  function_space: fd.FunctionSpace
  solver_parameters: dict = None
  copy_output: bool = False

  def __post_init__(self):
    self._f = fd.Function(self.function_space)
    self._v = fd.Function(self.function_space)
    if self.solver_parameters is None:
      self.solver_parameters = MUMPS_SOLVER_PARAMETERS

    self._problem = fd.LinearVariationalProblem(
        self.J, self.M(self._v), self._f, bcs=self.bcs, constant_jacobian=True)
    self._solver = fd.LinearVariationalSolver(
        self._problem, solver_parameters=self.solver_parameters)

  @property
  def T(self) -> InverseOperator:
    """Return the adjoint operator.

        This will solve the matrix pencil A^T @ f = M^T @ v0 for f.
        """
    cls = self.__class__
    args = self.J.arguments()
    JT = ufl.adjoint(self.J, reordered_arguments=(args[0], args[1]))
    return cls(JT, self.M, self.bcs, self.function_space,
               self.solver_parameters)

  def __matmul__(self, v0):
    """Solve the matrix pencil A @ f = M @ v0 for f.

        This is equivalent to the "inverse iteration" f = (A^{-1} @ M) @ v0
        """
    self._v.assign(v0)
    self._solver.solve()
    if self.copy_output:
      return self._f.copy(deepcopy=True)
    return self._f


def adjoint(L):
  args = L.arguments()
  L_adj = ufl.adjoint(L, reordered_arguments=(args[0], args[1]))
  return L_adj


def define_inner_product(mass_matrix):
  if isinstance(mass_matrix, sparse.csr_matrix):
    M = mass_matrix
  else:
    # Assume filename
    if mass_matrix[-4:] != ".npz":
      mass_matrix += ".npz"
    M = sparse.load_npz(mass_matrix)

  def inner_product(u, v):
    return np.dot(u.conj(), M @ v)

  return inner_product


def pod(
    flow: FlowConfig,
    snapshot_handles: Iterable[Snapshot],
    r: int,
    mass_matrix,
    decomp_indices=None,
    remove_mean=True,
    mean_dest="mean",
    atol=1e-13,
    rtol=None,
    max_vecs_per_node=None,
    verbosity=1,
    output_dir=".",
    modes_dest="pod",
    eigvals_dest="eigvals.dat",
    pvd_dest=None,
    coeffs_dest="coeffs.dat",
    field_name="q",
):
  """
    Args:
        ``flow``: ``FlowConfig`` for the POD (used for weighted inner product)

        ``snapshots``: List of Snapshot handles

        ``r``: Number of modes to compute

    Kwargs:
        ``decomp_indices``: Indices to use in method of snapshots (defaults to ``range(r)``)

    NOTE: could actually take the "flow" dependence out if we replaced the PVD with a postprocessing callback...
    """
  if fd.COMM_WORLD.size > 1:
    raise NotImplementedError("Not yet supported in parallel")

  # Compute actual POD
  if decomp_indices is None:
    decomp_indices = range(r)

  inner_product = define_inner_product(mass_matrix)

  if remove_mean:
    base_vec_handle = Snapshot(f"{output_dir}/{mean_dest}")
    base_vec_handle.put(vec_handle_mean(snapshot_handles))

    # Redefine snapshots with mean subtraction
    snapshot_handles = [
        Snapshot(snap.filename, base_vec_handle=base_vec_handle)
        for snap in snapshot_handles
    ]
    logging.log(logging.DEBUG, "Mean subtracted")

  logging.log(logging.DEBUG, "Computing POD")
  POD = PODHandles(
      inner_product=inner_product,
      max_vecs_per_node=max_vecs_per_node,
      verbosity=verbosity,
  )
  POD.compute_decomp(snapshot_handles, atol=atol, rtol=rtol)

  # Vector handles for storing snapshots
  mode_handles = [
      Snapshot(filename=f"{output_dir}/{modes_dest}{i}") for i in range(r)
  ]
  POD.compute_modes(range(r), mode_handles)

  POD.put_eigvals(f"{output_dir}/{eigvals_dest}")

  # Save for visualization
  if pvd_dest is not None:
    pvd = fd.File(f"{output_dir}/{pvd_dest}", "w")
    for i, mode in enumerate(mode_handles):
      u, p = mode.get().as_function().subfunctions
      pvd.write(u, p, flow.vorticity(u))

  # Compute temporal coefficients
  coeffs = POD.compute_proj_coeffs().T  # If all snapshots used for POD
  np.savetxt(f"{output_dir}/{coeffs_dest}", coeffs, fmt="%0.6f", delimiter="\t")

  return coeffs, mode_handles


def project(basis_handles, data_handles, mass_matrix):
  inner_product = define_inner_product(mass_matrix)
  vec_space = VectorSpaceHandles(inner_product)
  coeffs = vec_space.compute_inner_product_array(basis_handles, data_handles).T
  return coeffs
