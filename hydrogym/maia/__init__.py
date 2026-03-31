"""
HydroGym Maia: Reinforcement Learning Environments for CFD Simulations
=======================================================================

HydroGym Maia provides OpenAI Gym compatible environments for computational fluid dynamics
simulations using the m-AIA solver, with integrated support for Hugging Face Hub data management.

Basic Usage (HPC/MPMD Mode):
    Step 1 - Prepare workspace (before submitting HPC job):
    >>> import hydrogym.maia as maia
    >>> work_dir, props_file = maia.prepare_maia_workspace('Cylinder_2D_Re200')

    Step 2 - Use in your SLURM/PBS job script:
    >>> # cd work_dir && srun --multi-prog mpmd.conf
    >>> # where mpmd.conf specifies Python and maia processes

Alternative Usage (Direct - for testing):
    >>> import hydrogym
    >>> env = hydrogym.maia.from_hf('Cylinder_2D_Re200')
    >>> obs, info = env.reset()
    >>> obs, reward, done, _, info = env.step(action)

Available Functions:
    - from_hf: Create any environment from Hugging Face Hub
    - prepare_maia_workspace: Setup workspace for MPMD coupling
    - list_available_environments: List all environments in HF repo
    - list_registered_types: Show all registered environment types
"""

# ── Always available — no MPI required ────────────────────────────────────────
# Safe to import standalone (e.g. prepare_workspace.py outside of mpirun).
from .hf_data_manager import HFDataManager
from .workspace import MaiaWorkspace, prepare_maia_workspace

# ── MPI-dependent names ────────────────────────────────────────────────────────
# Loaded lazily on first access via __getattr__ so that importing this package
# does NOT call `import mpi4py` unless an MPI-dependent symbol is actually used.
# This prevents mpi4py from hanging when the package is imported outside mpirun
# (e.g. for workspace preparation on a login node).

_MPI_ATTRS = frozenset(
    [
        "MaiaFlowEnv",
        "ConfigError",
        "from_hf",
        "list_available_environments",
        "list_registered_types",
        "MaiaInterface",
        "Cylinder",
        "RotaryCylinder",
        "Cavity",
        "Cavity3Jet",
        "Pinball",
        "JetPinball",
        "NACA0012",
        "NACA0012Gust",
        "SquareCylinder",
        "Cube",
        "Sphere",
        "ZPGTBLBase",
        "ZPGTBLJet",
        "ZPGTBLSurfaceWave",
        "DRA2303Base",
        "DRA2303Jet",
        "DRA2303SurfaceWave",
    ]
)


def _load_mpi_deps() -> None:
    """Import all MPI-dependent components and inject them into this module."""
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        raise ImportError(
            "MAIA solver dependencies are not installed. Install them with: pip install hydrogym[maia]"
        ) from None

    import sys

    _mod = sys.modules[__name__]

    from .env_core import (
        MaiaFlowEnv,
        ConfigError,
        from_hf,
        list_available_environments,
        list_registered_types,
    )
    from .mpmd_interface import MaiaInterface
    from .envs.cylinder import Cylinder, RotaryCylinder
    from .envs.cavity import Cavity, Cavity3Jet
    from .envs.pinball import Pinball, JetPinball
    from .envs.naca0012 import NACA0012, NACA0012Gust
    from .envs.square_cylinder import SquareCylinder
    from .envs.cube import Cube
    from .envs.sphere import Sphere
    from .envs.turbulent_boundary_layer import ZPGTBLBase, ZPGTBLJet, ZPGTBLSurfaceWave
    from .envs.dra2303 import DRA2303Base, DRA2303Jet, DRA2303SurfaceWave

    for _name, _obj in [
        ("MaiaFlowEnv", MaiaFlowEnv),
        ("ConfigError", ConfigError),
        ("from_hf", from_hf),
        ("list_available_environments", list_available_environments),
        ("list_registered_types", list_registered_types),
        ("MaiaInterface", MaiaInterface),
        ("Cylinder", Cylinder),
        ("RotaryCylinder", RotaryCylinder),
        ("Cavity", Cavity),
        ("Cavity3Jet", Cavity3Jet),
        ("Pinball", Pinball),
        ("JetPinball", JetPinball),
        ("NACA0012", NACA0012),
        ("NACA0012Gust", NACA0012Gust),
        ("SquareCylinder", SquareCylinder),
        ("Cube", Cube),
        ("Sphere", Sphere),
        ("ZPGTBLBase", ZPGTBLBase),
        ("ZPGTBLJet", ZPGTBLJet),
        ("ZPGTBLSurfaceWave", ZPGTBLSurfaceWave),
        ("DRA2303Base", DRA2303Base),
        ("DRA2303Jet", DRA2303Jet),
        ("DRA2303SurfaceWave", DRA2303SurfaceWave),
    ]:
        setattr(_mod, _name, _obj)


def __getattr__(name: str):
    if name in _MPI_ATTRS:
        _load_mpi_deps()
        import sys

        try:
            return getattr(sys.modules[__name__], name)
        except AttributeError:
            pass
    raise AttributeError(f"module 'hydrogym.maia' has no attribute {name!r}")


__all__ = [
    # Always available
    "HFDataManager",
    "MaiaWorkspace",
    "prepare_maia_workspace",
    # MPI-dependent (lazy)
    "MaiaFlowEnv",
    "ConfigError",
    "from_hf",
    "list_available_environments",
    "list_registered_types",
    "MaiaInterface",
    "Cylinder",
    "RotaryCylinder",
    "Cavity",
    "Cavity3Jet",
    "Pinball",
    "JetPinball",
    "NACA0012",
    "NACA0012Gust",
    "SquareCylinder",
    "Cube",
    "Sphere",
    "ZPGTBLBase",
    "ZPGTBLJet",
    "ZPGTBLSurfaceWave",
    "DRA2303Base",
    "DRA2303Jet",
    "DRA2303SurfaceWave",
]
