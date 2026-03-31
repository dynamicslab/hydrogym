# Lazy imports to avoid MPI initialization conflicts in MPMD mode
# Only import the core classes by default
from .core import CallbackBase, FlowEnv, PDEBase, TransientSolver


# Make submodules available via getattr for lazy loading
def __getattr__(name):
  """Lazy import submodules to avoid MPI conflicts."""
  import importlib

  if name in ("distributed", "firedrake", "maia", "nek"):
    # Use importlib to import the submodule
    module = importlib.import_module(f".{name}", package=__name__)
    # Cache it in globals to avoid re-importing
    globals()[name] = module
    return module

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Explicitly list what's available at top level
__all__ = [
    "CallbackBase", "FlowEnv", "PDEBase", "TransientSolver", "distributed",
    "firedrake", "maia", "nek"
]
