"""
HydroGym Nek5000 backend.

Initialization Patterns:
1. MAIA pattern (recommended):
   env = NekEnv.from_hf('MiniChannel_Re180', nproc=10)

2. Legacy pattern (deprecated):
   conf = OmegaConf.load('config.yaml')
   env = NekEnv(conf=conf)

Three-layer architecture:
1. NekEnv - Base single-agent Gym environment (array-based)
2. NekParallelEnv - Dict-based multi-agent wrapper
3. NekPettingZooEnv - Optional PettingZoo compatibility layer

Most users should use NekEnv directly.
"""

from .AFC import AFC, BLCtrl, OppoCtrl, SinWave, ZeroCtrl, make_afc_controller
from .configs import Config
from .env import ConfigError, NekEnv, mpi_split
from .parallel_env import NekParallelEnv
from .utils import io
from .utils.utils import is_rank_zero, print

# Try to import PettingZoo wrapper (optional)
try:
    from .pettingzoo_env import NekPettingZooEnv, make_pettingzoo_env

    __all_pettingzoo__ = ["NekPettingZooEnv", "make_pettingzoo_env"]
except ImportError:
    __all_pettingzoo__ = []

# Import integrate last to avoid circular imports
from .integrate import integrate

__all__ = [
    # Core environment classes
    "NekEnv",
    "NekParallelEnv",
    # Configuration
    "Config",
    "ConfigError",
    # Utilities
    "mpi_split",
    "integrate",
    "io",
    "is_rank_zero",
    "print",
    # Controllers
    "AFC",
    "OppoCtrl",
    "BLCtrl",
    "SinWave",
    "ZeroCtrl",
    "make_afc_controller",
] + __all_pettingzoo__


# Backwards compatibility: provide old names
def load_nek_config(config_path: str, overrides=None):
    """
    Load Nek config from YAML and apply overrides.

    DEPRECATED: This function is kept for backwards compatibility.
    The new NekEnv accepts OmegaConf objects directly.
    """
    from omegaconf import OmegaConf

    if overrides is None:
        overrides = []

    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(config_path),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


# Legacy compatibility: NekMARLGymWrapper -> NekEnv
# This allows old code to still work
NekMARLGymWrapper = NekEnv

# Legacy: parallel_env -> NekParallelEnv
parallel_env = NekParallelEnv

__all__.extend(["load_nek_config", "NekMARLGymWrapper", "parallel_env"])
