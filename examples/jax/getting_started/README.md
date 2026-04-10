# Getting Started with JAX Environments

**START HERE** for JAX-based flow control examples using `env.reset_env()` and `env.step_env()`.

This directory contains three examples covering the two available JAX environments and a pure-JAX PPO training loop that works with both.

> **Note:** JAX environments run entirely on GPU inside a single JIT-compiled Python process — no MPI or separate solver process is needed.

## Directory Structure

Each subdirectory demonstrates a specific environment or training workflow:

### 1. [`1_kolmogorov/`](1_kolmogorov/) — 2D Kolmogorov Flow
**Solver:** 2D pseudo-spectral Navier-Stokes (64×64 Fourier modes, Re=200)
**Action:** 4 sinusoidal body-force amplitudes (wavenumbers 4–7)
**Observation:** 8×8 grid of velocity probe values (64-dimensional)
**Reward:** `-(reward_alpha * TKE + action_penalty)`
**Precision:** float64 required for JIT stability

```python
import jax
jax.config.update("jax_enable_x64", True)  # required for JIT stability

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})
params = env.default_params

# Tune reward objective via reward_alpha:
#   reward_alpha > 0  →  suppress TKE (laminarization)
#   reward_alpha < 0  →  maximize TKE (enhance mixing)
params = params.replace(reward_alpha=1.0)

jit_reset = jax.jit(env.reset_env)
jit_step  = jax.jit(env.step_env)

key = jax.random.PRNGKey(0)
obs, state = jit_reset(key, params)
action = jnp.array([-0.25, -0.03, 0.02, 0.01])  # 4 body-force amplitudes
obs, state, reward, done, info = jit_step(key, state, action, params)
```

**Files:**
- `test_kolmogorov_env.py` — JIT-compiled environment runner with three modes and configurable `--dtype`, `--dt`, `--num-steps`
- `kolmogorov.ipynb` — interactive notebook with vorticity visualizations
- `run_kolmogorov_docker.sh` — bash launcher: `run_kolmogorov_docker.sh [mode] [num_steps] [dtype]`

**Bash usage:**
```bash
./run_kolmogorov_docker.sh                            # minimize_tke, 10 steps, float64
./run_kolmogorov_docker.sh maximize_tke 500           # maximize TKE, 500 steps
./run_kolmogorov_docker.sh minimize_tke 100 float32   # float32 (may diverge — use smaller --dt)
```

**Precision note:** The pseudo-spectral 2D NS solver is numerically marginally stable. Under XLA compilation (JIT), float32 arithmetic reordering can trigger blow-up. Use float64 (default) for reliable results. To experiment with float32, reduce `dt` via `env_config={"dt": 5e-4}` or `--dt 5e-4`.

---

### 2. [`2_channel/`](2_channel/) — 3D Turbulent Channel Flow
**Solver:** 3D pseudo-spectral DNS (72×72×72, Re_tau=180)
**Action:** 24 wall-normal jet amplitudes on a 6×4 grid of Gaussian patches
**Observation:** 8×8×2 near-wall velocity samples (128-dimensional)
**Reward:** `-WSS` (minimize wall shear stress / skin-friction drag)
**Precision:** float32 by default (stable); float64 available

```python
import jax
from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

env = ChannelFlowSpectralEnv(env_config={"dtype": "float32"})  # or "float64"
params = env.default_params

jit_reset = jax.jit(env.reset_env)
jit_step  = jax.jit(env.step_env)

key = jax.random.PRNGKey(0)
obs, state = jit_reset(key, params)

# Zero action: passive turbulence evolution (baseline)
action = jnp.zeros((params.action_dim,))   # shape (24,)
obs, state, reward, done, info = jit_step(key, state, action, params)
# reward = -WSS;  WSS ≈ 0.0019 for uncontrolled Re_tau=180
```

**Note:** The initial turbulent field (`U.npy`, `V.npy`, `W.npy`) is downloaded from Hugging Face Hub on the first run and cached at `~/.cache/hydrogym/`.

**Files:**
- `test_channel_env.py` — JIT-compiled environment runner with three modes and configurable `--dtype`, `--num-steps`
- `channel.ipynb` — interactive notebook
- `run_channel_docker.sh` — bash launcher: `run_channel_docker.sh [mode] [num_steps] [dtype]`

**Bash usage:**
```bash
./run_channel_docker.sh                               # no_actuation, 5 steps, float32
./run_channel_docker.sh drag_reduction 20             # 20 steps
./run_channel_docker.sh strong_actuation 10 float64   # float64 precision
```

---

### 3. [`3_ppo/`](3_ppo/) — Pure-JAX PPO Training
**Interface:** Works with both `KolmogorovFlow` and `ChannelFlowSpectralEnv`
**Framework:** Pure JAX (Flax + Optax + Distrax) — no Stable-Baselines3
**Style:** PureJAX — entire rollout + update loop is JIT-compiled via `jax.lax.scan`

```python
# Select environment via --env argument
python run_ppo.py --env kolmogorov --total-timesteps 20000
python run_ppo.py --env channel    --total-timesteps 5000 --num-envs 2 --num-steps 5
```

The training loop uses `VecEnv` to run `NUM_ENVS` parallel environments inside a single JIT-compiled scan, which is significantly faster than Python-level parallelism.

**Key hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `--total-timesteps` | 4000 | Total environment steps |
| `--num-envs` | 4 | Parallel environments per update |
| `--num-steps` | 10 | Rollout length per environment |
| `--num-minibatches` | 8 | Must divide `num_envs × num_steps` |
| `--lr` | 1e-4 | Adam learning rate |

**Output:**
- `trained_model_<env>.pkl` — serialized Flax parameter tree
- `plot_reward_<env>.png` — episode return vs. update steps
- `rewardovertime.npy` — raw return array

**Files:**
- `run_ppo.py` — full PPO implementation (ActorCritic, GAE, clipped surrogate loss)
- `run_ppo_docker.sh` — bash wrapper that activates the venv and forwards all arguments

---

## Quick Start

### 1. Choose Your Example

- **Explore flow physics?** → Start with `1_kolmogorov/` (faster, 2D)
- **Drag reduction?** → Try `2_channel/` (more expensive, 3D DNS)
- **Train an RL agent?** → Go to `3_ppo/`

### 2. Test the Environment

```bash
cd 1_kolmogorov/
./run_kolmogorov_docker.sh                        # minimize TKE, 10 steps, float64
./run_kolmogorov_docker.sh maximize_tke           # maximize TKE
./run_kolmogorov_docker.sh no_actuation 500       # baseline, 500 steps
./run_kolmogorov_docker.sh minimize_tke 10 float32  # float32 (may diverge)
```

```bash
cd 2_channel/
./run_channel_docker.sh                           # baseline (no actuation), float32
./run_channel_docker.sh drag_reduction 20         # uniform suction, 20 steps
./run_channel_docker.sh strong_actuation 10 float64  # checkerboard jets, float64
```

### 3. Train a PPO Agent

```bash
cd 3_ppo/
./run_ppo_docker.sh --env kolmogorov --total-timesteps 20000
./run_ppo_docker.sh --env channel --num-envs 2 --num-steps 5 --total-timesteps 5000
./run_ppo_docker.sh --help   # full argument list
```

## Comparison Table

| Directory | Environment | Dims | Action | Precision | GPU cost | Best For |
|---|---|---|---|---|---|---|
| **1_kolmogorov** | `KolmogorovFlow` | 2D 64×64 | 4 body-force modes | float64 | Low | Quick experiments, tunable reward |
| **2_channel** | `ChannelFlowSpectralEnv` | 3D 72³ | 24 wall jets | float32 | High | Turbulent drag reduction |
| **3_ppo** | Both | — | — | Env-dependent | Depends on env | End-to-end JAX RL training |

## Requirements

- JAX with GPU support (`jax[cuda12]` or equivalent)
- `flax`, `optax`, `distrax` (for `3_ppo/`)
- Internet access on first run for the channel flow initial field

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
