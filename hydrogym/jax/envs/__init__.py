"""JAX environments.

========================== ============================================ =========================================
Name                       Module                                       Description
========================== ============================================ =========================================
``ChannelFlowEnv``         ``hydrogym.jax.envs.channel_fd``             factory: ``env_config["solver"]`` is
                                                                        ``"fd"`` (default) or ``"spectral"``
``ChannelFlowFDEnv``       ``hydrogym.jax.envs.channel_fd``             finite-difference channel, ``Re_tau = 180``
                                                                        (default channel)
``ChannelFlowSpectralEnv`` ``hydrogym.jax.envs.channel``                pseudo-spectral channel (legacy; its flow
                                                                        runs at ``Re_tau`` of about 310)
``KolmogorovFlow``         ``hydrogym.jax.envs.kolmogorov``             two-dimensional Kolmogorov flow
========================== ============================================ =========================================

The classes are imported on first access, so that importing this package does not load a solver module (the
pseudo-spectral channel module switches on 64-bit precision for the whole process when it is imported).
"""

import importlib

_EXPORTS = {
    "ChannelFlowEnv": "hydrogym.jax.envs.channel_fd",
    "ChannelFlowFDEnv": "hydrogym.jax.envs.channel_fd",
    "ChannelFlowSpectralEnv": "hydrogym.jax.envs.channel",
    "KolmogorovFlow": "hydrogym.jax.envs.kolmogorov",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
