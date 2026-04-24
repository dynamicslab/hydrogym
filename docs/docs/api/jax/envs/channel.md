---
sidebar_label: channel
title: hydrogym.jax.envs.channel
---

## ChannelEnvParams Objects

```python
@struct.dataclass
class ChannelEnvParams(BaseEnvParams)
```

Extends base EnvParams with channel-specific settings.

#### nsteps

DNS substeps per RL step

## SpectralState Objects

```python
class SpectralState(NamedTuple)
```

#### u\_hat

(Nx,Ny,Nz), complex

## ChannelFlowSpectralEnv Objects

```python
class ChannelFlowSpectralEnv(JAXFlowEnvBase)
```

3D turbulent channel flow environment using a pseudo-spectral DNS solver.

