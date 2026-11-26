# Lazy imports to avoid MPI initialization conflicts in MPMD mode
# Only import the core classes by default
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core import CallbackBase, FlowEnv, PDEBase, TransientSolver

try:
    __version__ = _version("hydrogym")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "1.0.2"


# Make submodules available via getattr for lazy loading
def __getattr__(name):
    """Lazy import submodules to avoid MPI conflicts."""
    import importlib

    if name in ("firedrake", "jax", "jaxfluids", "maia", "nek", "nekrs"):
        # Use importlib to import the submodule
        module = importlib.import_module(f".{name}", package=__name__)
        # Cache it in globals to avoid re-importing
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Explicitly list what's available at top level
__all__ = [
    "CallbackBase",
    "FlowEnv",
    "PDEBase",
    "TransientSolver",
    "firedrake",
    "jax",
    "jaxfluids",
    "maia",
    "nek",
    "nekrs",
]
