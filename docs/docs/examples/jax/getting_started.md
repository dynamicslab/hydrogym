---
sidebar_position: 1
---

# Getting Started

Examples demonstrating HydroGym's JAX backend for GPU-accelerated, fully-differentiable flow control.

## What is the JAX backend?

HydroGym's JAX backend provides pseudo-spectral Navier-Stokes solvers written entirely in JAX. This enables:

- **GPU acceleration** — solvers run on GPU via JAX's XLA compilation
- **Vectorized environments** — run many parallel environments inside a single JIT-compiled training loop (PureJAX-style)
- **End-to-end differentiability** — gradients can flow through the solver for gradient-based control

The JAX environments follow the [gymnax](https://github.com/RobertTLange/gymnax) interface (`reset_env` / `step_env` with explicit `params`) and include wrappers (`VecEnv`, `LogWrapper`, `ClipAction`, `NormalizeVecObservation`, `NormalizeVecReward`) for RL training.

## Directory Structure

```
examples/jax/
├── README.md                    # Package overview
└── getting_started/             # START HERE
    ├── README.md                # Detailed guide and comparison table
    ├── 1_kolmogorov/            # 2D Kolmogorov flow (Re=200)
    ├── 2_channel/               # 3D turbulent channel flow (Re_tau=180)
    └── 3_ppo/                   # Pure-JAX PPO training (both environments)
```

## Quick Start

```bash
# Activate the GPU environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

# Test Kolmogorov flow
cd getting_started/1_kolmogorov
./run_kolmogorov_docker.sh

# Test channel flow
cd getting_started/2_channel
./run_channel_docker.sh

# Train PPO
cd getting_started/3_ppo
./run_ppo_docker.sh --env kolmogorov --total-timesteps 20000
```

## Available Environments

| Environment | Solver | Grid | Action | Observation | Reward |
|---|---|---|---|---|---|
| `KolmogorovFlow` | 2D pseudo-spectral | 64×64 | 4 body-force modes | 8×8 velocity probes | -(α·TKE + action penalty) |
| `ChannelFlowSpectralEnv` | 3D pseudo-spectral | 72×72×72 | 24 wall jets | 8×8×2 near-wall velocities | -WSS (drag) |

## Typical Usage

```python
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})
params = env.default_params

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

action = jnp.zeros((params.action_dim,))
obs, state, reward, done, info = env.step_env(key, state, action, params)
```

**Note:** The channel flow environment downloads a fully turbulent initial field from Hugging Face Hub (`dynamicslab/HydroGym-environments`) on the first run and caches it at `~/.cache/hydrogym/`.

## Requirements

- JAX with GPU support (`jax[cuda12]` or equivalent)
- `flax`, `optax`, `distrax` for PPO training
- Internet access on first run (channel flow initial field download)
