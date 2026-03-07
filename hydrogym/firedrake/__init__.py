try:
  import firedrake  # noqa: F401
except ImportError:
  raise ImportError(
      "Firedrake is not installed. "
      "Install it via its own installer: https://www.firedrakeproject.org/download.html\n"
      "Also install mesh generation support with: pip install hydrogym[firedrake]"
  ) from None

from hydrogym.core import FlowEnv

from .actuator import DampedActuator
from .flow import FlowConfig, ObservationFunction, ScaledDirichletBC
from .solvers import LinearizedBDF, NewtonSolver, SemiImplicitBDF, integrate
from .utils import io, is_rank_zero, linalg, modeling, print

from .envs import Cylinder, RotaryCylinder, Pinball, Cavity, Step  # isort:skip

__all__ = [
    "FlowConfig",
    "DampedActuator",
    "ScaledDirichletBC",
    "ObservationFunction",
    "NewtonSolver",
    "SemiImplicitBDF",
    "integrate",
    "FlowEnv",
    "Cylinder",
    "RotaryCylinder",
    "Pinball",
    "Cavity",
    "Step",
    "io",
    "is_rank_zero",
    "linalg",
    "modeling",
    "print",
]
