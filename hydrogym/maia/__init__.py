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

# Import core classes and exceptions
from .env_core import (
    MaiaFlowEnv,
    ConfigError,
    from_hf,
    list_available_environments,
    list_registered_types,
)

# Import HF data manager
from .hf_data_manager import HFDataManager

# Import MPI interface
from .mpmd_interface import MaiaInterface

# Import workspace utilities
from .workspace import MaiaWorkspace, prepare_maia_workspace

# Import HF-enabled environments (this registers them automatically)
from .envs.cylinder import Cylinder, RotaryCylinder
from .envs.cavity import Cavity, Cavity3Jet
from .envs.pinball import Pinball, JetPinball
from .envs.naca0012 import NACA0012, NACA0012Gust
from .envs.square_cylinder import SquareCylinder
from .envs.cube import Cube
from .envs.sphere import Sphere

# Make the factory function available at package level
__all__ = [
    # Core classes
    'MaiaFlowEnv',
    'ConfigError',

    # Factory functions
    'from_hf',
    'list_available_environments',
    'list_registered_types',

    # Data management
    'HFDataManager',

    # MPI interface
    'MaiaInterface',

    # Workspace utilities
    'MaiaWorkspace',
    'prepare_maia_workspace',

    # HF-enabled environments
    'Cylinder',
    'RotaryCylinder',
    'Cavity',
    'Cavity3Jet',
    'Pinball',
    'JetPinball',
    'NACA0012',
    'NACA0012Gust',
    'SquareCylinder',
    'Cube',
    'Sphere'
]
