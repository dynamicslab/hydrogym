from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from scipy import sparse

from hydrogym.utils import DependencyNotInstalled

import firedrake as fd
import ufl
from firedrake import logging


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
            self.J, self.M(self._v), self._f, bcs=self.bcs, constant_jacobian=True
        )
        self._solver = fd.LinearVariationalSolver(self._problem, solver_parameters=self.solver_parameters)

    @property
    def T(self) -> InverseOperator:
        """Return the adjoint operator.

        This will solve the matrix pencil A^T @ f = M^T @ v0 for f.
        """
        cls = self.__class__
        args = self.J.arguments()
        JT = ufl.adjoint(self.J, reordered_arguments=(args[0], args[1]))
        return cls(JT, self.M, self.bcs, self.function_space, self.solver_parameters)

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
