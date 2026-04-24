# JAX Examples

Examples demonstrating HydroGym's JAX backend for GPU-accelerated, fully-differentiable flow control.

## What is the JAX backend?

HydroGym's JAX backend provides pseudo-spectral Navier-Stokes solvers written entirely in JAX. This enables:

- **GPU acceleration** — solvers run on GPU via JAX's XLA compilation
- **Vectorized environments** — run many parallel environments inside a single JIT-compiled training loop (PureJAX-style)
- **End-to-end differentiability** — gradients can flow through the solver for gradient-based control

The JAX environments follow the [gymnax](https://github.com/RobertTLange/gymnax) interface (`reset_env` / `step_env` with explicit `params`) and include wrappers (`VecEnv`, `LogWrapper`, `ClipAction`, `NormalizeVecObservation`, `NormalizeVecReward`) for RL training.

## Directory Structure

```
jax/
├── README.md                    # This file
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

# Test Kolmogorov flow (float64, 10 steps)
cd getting_started/1_kolmogorov
./run_kolmogorov_docker.sh

# Test channel flow (float32, 5 steps)
cd getting_started/2_channel
./run_channel_docker.sh

# Train PPO
cd getting_started/3_ppo
./run_ppo_docker.sh --env kolmogorov --total-timesteps 20000
```

## Available Environments

| Environment | Solver | Grid | Action | Observation | Reward | Default dtype |
|---|---|---|---|---|---|---|
| `KolmogorovFlow` | 2D pseudo-spectral | 64×64 | 4 body-force modes | 8×8 velocity probes | -(α·TKE + action penalty) | float64 |
| `ChannelFlowSpectralEnv` | 3D pseudo-spectral | 72×72×72 | 24 wall jets | 8×8×2 near-wall velocities | -WSS (drag) | float32 |

## JIT Compilation

Both environments are JIT-compiled via `jax.jit` in the runner scripts, which compiles the full DNS rollout into a single GPU kernel:

```python
jit_reset = jax.jit(env.reset_env)
jit_step  = jax.jit(env.step_env)

obs, state = jit_reset(key, params)           # triggers compilation
obs, state, reward, done, info = jit_step(key, state, action, params)  # full GPU speed
```

The first call compiles (takes ~1–2 minutes); all subsequent calls run at full GPU speed.

## Floating-Point Precision

| Environment | Recommended | Notes |
|---|---|---|
| `KolmogorovFlow` | `float64` | Pseudo-spectral 2D NS requires fp64 for JIT stability; fp32 may produce NaNs under XLA reordering |
| `ChannelFlowSpectralEnv` | `float32` | Stable at fp32 with JIT; fp64 available but ~2x slower on A100 |

Override via `env_config`:
```python
# Kolmogorov: float64 is the default and required for JIT stability
env = KolmogorovFlow(env_config={"dt": 5e-4})          # smaller dt for fp32 experiments

# Channel: toggle precision
env = ChannelFlowSpectralEnv(env_config={"dtype": "float64"})
```

Or via the bash scripts:
```bash
./run_kolmogorov_docker.sh minimize_tke 100 float32   # float32 (may diverge)
./run_channel_docker.sh drag_reduction 10 float64     # float64
```

## Typical Usage

```python
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

jax.config.update("jax_enable_x64", True)  # required for Kolmogorov + JIT

env = KolmogorovFlow(env_config={}, flow_config={})
params = env.default_params

jit_reset = jax.jit(env.reset_env)
jit_step  = jax.jit(env.step_env)

key = jax.random.PRNGKey(0)
obs, state = jit_reset(key, params)

action = jnp.zeros((params.action_dim,))
obs, state, reward, done, info = jit_step(key, state, action, params)
```

**Note:** The channel flow environment downloads a fully turbulent initial field from Hugging Face Hub (`dynamicslab/HydroGym-environments`) on the first run and caches it at `~/.cache/hydrogym/`.

## Requirements

- JAX with GPU support (`jax[cuda12]` or equivalent)
- `flax`, `optax`, `distrax` for PPO training
- Internet access on first run (channel flow initial field download)
