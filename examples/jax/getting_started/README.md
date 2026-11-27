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
**Precision:** float32 or float64 (the example defaults to float64)

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64; leave it off for float32

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})
params = env.default_params

# Tune reward objective via reward_alpha:
#   reward_alpha > 0  →  suppress TKE (laminarization)
#   reward_alpha < 0  →  maximize TKE (enhance mixing)
params = params.replace(reward_alpha=1.0)

jit_reset = jax.jit(env.reset_env)
jit_step = jax.jit(env.step_env)

key = jax.random.PRNGKey(0)
obs, state = jit_reset(key, params)
action = jnp.array([-0.25, -0.03, 0.02, 0.01])  # 4 body-force amplitudes
obs, state, reward, done, info = jit_step(key, state, action, params)
```

**Files:**
- `test_kolmogorov_env.py` — JIT-compiled environment runner with three modes and configurable `--dtype`, `--dt`, `--num-steps`
- `kolmogorov.ipynb` — interactive notebook with vorticity visualizations
- `run_kolmogorov_docker.sh` — bash launcher: `run_kolmogorov_docker.sh [mode] [num_steps] [dtype]` (in the HydroGym
  images, run it from a copy of the examples, `cp -r /opt/hydrogym/examples ~/hydrogym-examples`, unless you are
  root: it writes into its directory)

**Bash usage:**
```bash
./run_kolmogorov_docker.sh                            # minimize_tke, 10 steps, float64
./run_kolmogorov_docker.sh maximize_tke 500           # maximize TKE, 500 steps (float64: about 45-60 min on one GH200)
./run_kolmogorov_docker.sh minimize_tke 100 float32   # float32
```

**Precision note:** The solver runs in float32 and in float64 (set by `jax_enable_x64`). float32 is faster and uses
half the memory; the released extreme-event preset and its policies use it. The flow is chaotic, so float32 and float64
trajectories separate after a few actions. The DNS time step is set with
`env_config={"dt": ...}` or `--dt` (default 1e-3).

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
jit_step = jax.jit(env.step_env)

key = jax.random.PRNGKey(0)
obs, state = jit_reset(key, params)

# Zero action: passive turbulence evolution (baseline)
action = jnp.zeros((params.action_dim,))  # shape (24,)
obs, state, reward, done, info = jit_step(key, state, action, params)
# reward = -WSS. The magnitude is configuration-dependent (normalization,
# grid, initial field); for the uncontrolled case at Re_tau=180 WSS is O(0.3).
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

```bash
# Select environment via --env argument; the defaults are getting-started settings
python run_ppo.py --env kolmogorov   # ~15-20 min on a laptop CPU
python run_ppo.py --env channel      # ~15-20 min on a laptop CPU
```

The training loop uses `VecEnv` to run `NUM_ENVS` parallel environments inside a single JIT-compiled scan, which is significantly faster than Python-level parallelism.

**Getting-started settings.** The environments' own episodes are long: Kolmogorov runs 1000 actions of 10 time units
(10,000 solver steps of dt = 1e-3 per action), the channel 100 actions of 50 DNS steps (`max_steps_in_episode = 5000`
counts DNS steps). With them, a CPU run that ends a few episodes takes hours to a day. The defaults of `run_ppo.py`
therefore shorten the episodes so that episodes end within minutes:

| | Kolmogorov | Channel |
|---|---|---|
| Episode | 20 actions of 1 time unit, dt = 5e-3 (200 solver steps per action) | 10 actions of 50 DNS steps |
| `--total-timesteps` / `--num-envs` / `--num-steps` / `--num-minibatches` | 4000 / 4 / 20 / 4 | 120 / 2 / 10 / 2 |
| PPO updates, episodes per environment | 50, 50 | 6, 6 |

dt = 5e-3 changes the time-averaged TKE by less than 0.5 % compared with dt = 1e-3. The Kolmogorov reward is
`-(reward_alpha * TKE + sum |a|)`; the environment's default `reward_alpha = 0` leaves only the action penalty, so
`run_ppo.py` sets `--reward-alpha 1` (reduce TKE at a small actuation cost).

**Key hyperparameters** (both environments unless noted):

| Parameter | Default | Description |
|---|---|---|
| `--env-defaults` | off | Use the environment's own episode length and action window |
| `--episode-steps` | 20 / 10 | Actions per episode (Kolmogorov / channel) |
| `--action-time`, `--dt` | 1, 5e-3 | Kolmogorov: time units per action, solver time step |
| `--reward-alpha` | 1 | Kolmogorov: TKE weight in the reward |
| `--total-timesteps` | 4000 / 120 | Total environment steps, all environments together |
| `--num-envs` | 4 / 2 | Parallel environments per update |
| `--num-steps` | 20 / 10 | Rollout length per environment and update |
| `--num-minibatches` | 4 / 2 | Must divide `num_envs × num_steps` |
| `--lr` | 1e-3 | Adam learning rate |
| `--initial-log-std` | -1 | Initial log standard deviation of the policy (std 0.37) |
| `--gamma`, `--gae-lambda` | 0.5, 0.9 | Discount factor, GAE lambda (use e.g. 0.99 for long runs) |
| `--clip-value` / `--no-clip-value` | on | Clip the value update to 0.2 (purejaxrl) |
| `--ent-coef` | 0 | Entropy bonus |

Full-size runs (GPU recommended) keep the environments' episodes, for example

```bash
python run_ppo.py --env kolmogorov --env-defaults --total-timesteps 20000 --num-steps 100   # about a day on a CPU, ~12 h on an RTX A5000
python run_ppo.py --env channel --env-defaults --total-timesteps 5000 --num-steps 50        # about half a day on a CPU
```

The PPO defaults are chosen for the small getting-started budget: with a few thousand samples of a chaotic flow, a
short horizon (`gamma = 0.5`) and a small initial exploration noise (`--initial-log-std -1`) make the advantages
reflect the immediate effect of an action. With long-horizon settings (`--initial-log-std 0 --gamma 0.99
--gae-lambda 0.985 --ent-coef 0.1`) the getting-started Kolmogorov returns stay flat; for full-size runs, a larger
`--gamma` is the usual choice.

**What to expect.** Kolmogorov (about 15-20 min on a laptop CPU): over the 50 updates
the mean reward per step rises from about -2.5 to -1.8 and the episode return from about -50 to -35. Most of the gain
comes from the policy lowering its exploration noise (standard deviation from 0.37 to about 0.15) and with it the
actuation cost; the mean TKE drops only a little (about 1.38 to 1.29). Channel (about 15-20 min): the returns stay
flat at about -3.42 (differences below 0.2 %; the autoscaled axis magnifies them). Every episode starts from the same stored turbulent field, and
within 10 actions (500 DNS steps) the wall shear stress hardly responds to the jets, so this run only shows the
training pipeline end to end; drag reduction needs the full 100-action episodes and far more timesteps (GPU).

**Output:**
- `trained_model_<env>.pkl` — serialized Flax parameter tree
- `plot_reward_<env>.png` — mean reward per step for each PPO update with its running mean (top) and the returns of the episodes that ended in each update, every episode as a dot and their mean as a line (bottom)
- `rewardovertime.npy` — mean reward per step for each PPO update

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
./run_kolmogorov_docker.sh minimize_tke 10 float32  # float32
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
./run_ppo_docker.sh --env kolmogorov   # getting-started settings, ~15-20 min on a CPU
./run_ppo_docker.sh --env channel      # getting-started settings, ~15-20 min on a CPU
./run_ppo_docker.sh --help             # full argument list
```

## Comparison Table

| Directory | Environment | Dims | Action | Precision | GPU cost | Best For |
|---|---|---|---|---|---|---|
| **1_kolmogorov** | `KolmogorovFlow` | 2D 64×64 | 4 body-force modes | float64 | Low | Quick experiments, tunable reward |
| **2_channel** | `ChannelFlowSpectralEnv` (legacy channel, Re_τ ≈ 310; `ChannelFlowEnv` selects the finite-difference Re_τ = 180 channel) | 3D 72³ | 24 wall jets | float32 | High | Turbulent drag reduction |
| **3_ppo** | Both | — | — | Env-dependent | Depends on env | End-to-end JAX RL training |

## Requirements

- JAX with GPU support (`jax[cuda12]` or equivalent)
- `flax`, `optax`, `distrax` (for `3_ppo/`)
- Internet access on first run for the channel flow initial field

---

**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
