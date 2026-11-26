# Getting Started with NEK5000 Environments

**START HERE** for NEK5000-based RL interface examples using `env.reset()` and `env.step()`.

This directory contains comprehensive examples for using HydroGym's NEK5000-based flow environments with different interface patterns, from single-agent to multi-agent reinforcement learning.

> **Note:** NEK5000 requires MPI for parallel execution. All examples use `mpirun` to coordinate between the Python controller and NEK5000 solver processes.

> **Note:** The examples write into their directories. In the HydroGym images, `/opt/hydrogym/examples` is writable only for root: as any other user (and under Apptainer), copy the examples first, `cp -r /opt/hydrogym/examples ~/hydrogym-examples`, and run them from the copy.

> **Note:** The `run_*_docker.sh` scripts download the environment from the Hugging Face Hub. On compute nodes without
> internet access, download the environment folder (e.g. `TCFmini_3D_Re180/`) on a login node and point the scripts
> at the directory that holds it: `LOCAL_DIR=/path/to/envs ./run_nekenv_docker.sh` (default: `packaged_envs/` next to
> the script). Run the examples from a short path: Nek5000 accepts case paths of fewer than 132 characters.

## Standard entry point: `gym.make()`

For the common single-agent case, the standard, recommended way to start
a Nek5000 environment is gymnasium's `gym.make()` — same call shape as
every other backend, MPMD coupling handled underneath:

```bash
# from a prepared workspace (see prepare_workspace.py)
mpirun -np 1 python gym_make_demo.py --nproc 10 : -np 10 nek5000
# offline (compute node without internet): the directory given to prepare_workspace.py --local-dir
mpirun -np 1 python gym_make_demo.py --nproc 10 --local-dir /path/to/packaged_envs : -np 10 nek5000
```

```python
import gymnasium as gym
import hydrogym.registration

env = gym.make("hydrogym-nek/TCFmini_3D_Re180-v0", nproc=10)  # nproc required, must match the MPMD launch
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

See `gym_make_demo.py` in this directory. The 6 patterns below (multi-agent
wrappers, `from_hf`, `integrate()`, zero-shot deployment) are the
lower-level entry points for everything `gym.make()`'s single-agent
default doesn't cover — construct `NekEnv` directly, same as `gym.make()`
does underneath.

## Directory Structure

Each subdirectory demonstrates a specific interface pattern with complete examples:

### 1. [`1_nekenv_single/`](1_nekenv_single/) - Single Agent (Standard Gym)
**Interface:** `NekEnv` - Standard Gymnasium single-agent interface
**Use Case:** Single actuator/sensor scenarios
**SB3 Compatible:** ✅ Direct (no wrapper needed)

```python
from hydrogym.nek import NekEnv

env_config = {
    "environment_name": "TCFmini_3D_Re180",
    "nproc": 10,
}
env = NekEnv(env_config=env_config)

# Standard Gym interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Works directly with Stable-Baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

**Files:**
- `test_nek_direct.py` - Basic environment test with opposition control
- `train_sb3_nek_direct.py` - SB3 training with Monitor & VecNormalize
- `run_nekenv_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

### 2. [`2_parallel_env/`](2_parallel_env/) - Multi-Agent Parallel (PettingZoo)
**Interface:** `parallel_env` - PettingZoo parallel multi-agent
**Use Case:** Multiple independent agents with simultaneous actions
**SB3 Compatible:** ⚠️ Requires wrapper (SuperSuit or custom)

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.parallel_env import NekParallelEnv

# NekParallelEnv wraps an already-constructed NekEnv (composition, not a
# factory taking environment_name/nproc directly) -- one agent per actuator,
# discovered automatically from the base env's actuator info.
base_env = NekEnv(env_config={"environment_name": "TCFmini_3D_Re180", "nproc": 10})
env = NekParallelEnv(base_env)

# Dictionary-based observations and actions
obs, info = env.reset()  # {'jet_np...': array, ...}, one key per actuator agent
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

**Files:**
- `test_nek_parallel.py` - Multi-agent environment test
- `train_sb3_parallel.py` - SB3 training with SuperSuit wrappers
- `run_parallel_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

### 3. [`3_pettingzoo/`](3_pettingzoo/) - PettingZoo-Spec-Compliant Parallel API
**Interface:** `pettingzoo.ParallelEnv` subclass wrapping `NekParallelEnv`
**Use Case:** Same simultaneous, dict-based multi-agent interaction as
`2_parallel_env/` above, but as a genuine `pettingzoo.ParallelEnv`
subclass (PettingZoo `metadata`, cached `observation_space`/`action_space`)
for interop with PettingZoo-ecosystem tools that type-check for it.
**Not** the turn-based AEC (`agent_iter()`/`last()`) interface — Nek5000
has no AEC wrapper; every agent acts every step, same as `2_parallel_env/`.
**SB3 Compatible:** ⚠️ Requires wrapper

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv(env_config={"environment_name": "TCFmini_3D_Re180", "nproc": 10})
env = make_pettingzoo_env(base_env)

# Same dict-based, simultaneous-action API as NekParallelEnv above
obs, info = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

**Files:**
- `test_nek_pettingzoo.py` - `pettingzoo.ParallelEnv`-compliance test
- `train_sb3_pettingzoo.py` - Training with SuperSuit-wrapped parallel agents
- `run_pettingzoo_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

### 4. [`4_from_hf/`](4_from_hf/) - HuggingFace Packaged Environments
**Interface:** Load pre-packaged environments from HuggingFace Hub or local directories
**Use Case:** Using standardized, version-controlled environment configurations
**SB3 Compatible:** ✅ Works with any env type

```python
from hydrogym.nek import NekEnv

# Load a pre-packaged environment from HuggingFace Hub (or a local
# directory via local_fallback_dir) — workspace setup is automatic
env = NekEnv.from_hf(
    "TCFmini_3D_Re180",
    nproc=10,
    use_clean_cache=False,
    local_fallback_dir="./packaged_envs",  # optional
)

# Standard Gym interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Files:**
- `test_nek_DM.py` - from_hf() loading test
- `train_sb3_from_hf.py` - Training with HF environments
- `run_from_hf_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

### 5. [`5_hydrogym_control/`](5_hydrogym_control/) - HydroGym Controllers + Integrate
**Interface:** Using `hgym.integrate()` for time-stepping with controllers
**Use Case:** Classical control, RL deployment, or hybrid control strategies
**SB3 Compatible:** ✅ Pass trained model as controller

```python
from hydrogym.nek import NekEnv, integrate

# Train an RL agent
env = NekEnv(env_config=config)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)

# Use trained model as controller
integrate(
    env,
    t_span=(0, 100),
    controller=model,  # Can be trained model, PID, or custom controller
)
```

**Files:**
- `test_nek_env_controller.py` - Environment with controller test
- `train_sb3_with_integrate.py` - Training + deployment with integrate
- `run_control_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

### 6. [`6_zeroshot_wing_demo/`](6_zeroshot_wing_demo/) - Zero-Shot Wing Deployment
**Interface:** `NekEnv` + PettingZoo rollout with deployment controllers
**Use Case:** Deploy pre-trained DRL policies on the small wing without new training
**SB3 Compatible:** ✅ For loading trained policies; demo script is rollout-only

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)
```

**Files:**
- `zeroshot_demo_pettingzoo.py` - zero-shot wing rollout demo
- `meta_policy_small_wing_template.py` - template for multi-policy deployment
- `run_pettingzoo_docker.sh` - MPMD launch script (runs as is in the HydroGym images, from a copy of `/opt/hydrogym/examples` for a non-root user)

---

## Quick Start

### 1. Choose Your Interface
Pick the directory that matches your use case:
- **Single agent?** → Start with `1_nekenv_single/`
- **Multiple agents?** → Try `2_parallel_env/`
- **Pre-packaged environments?** → Use `4_from_hf/`
- **Deploy trained models?** → See `5_hydrogym_control/`
- **Zero-shot wing deployment?** → See `6_zeroshot_wing_demo/`

### 2. Test the Environment
```bash
cd 1_nekenv_single/
# from a prepared workspace: the Nek5000 case files must be in the working directory
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ws && cd ws
mpirun -np 1 python ../test_nek_direct.py --steps 100 : -np 10 nek5000
```

### 3. Train an RL Agent
```bash
# in the workspace ws/ of the block above
mpirun -np 1 python ../train_sb3_nek_direct.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## Comparison Table

| Directory | Interface | Obs Format | Action Format | SB3 Direct | Best For |
|-----------|-----------|------------|---------------|------------|----------|
| **1_nekenv_single** | `NekEnv` | Array | Array | ✅ Yes | Single actuator, simple baseline |
| **2_parallel_env** | `parallel_env` | Dict | Dict | ⚠️ Wrapper | Independent multi-agent scenarios |
| **3_pettingzoo** | `pettingzoo.ParallelEnv` | Dict | Dict | ⚠️ Wrapper | PettingZoo-ecosystem interop |
| **4_from_hf** | Any | Depends | Depends | Depends | Reproducible, versioned environments |
| **5_hydrogym_control** | Any + `integrate()` | Any | Any | ✅ Yes | Classical + RL hybrid control |
| **6_zeroshot_wing_demo** | PettingZoo Parallel | Dict | Dict | ✅ Deployment | Small-wing zero-shot DRL rollout |

## Requirements

### NEK5000 Setup
NEK5000 must be compiled and the `nek5000` executable must be in your PATH — the example launch lines invoke it directly as the second MPMD process group (`mpirun ... : -np 10 nek5000`), and there is no `nek_path`-style config key for it. We highly recommend using the HydroGym container images
(`clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` or `clagemann/hydrogym-cpu:1.0.2`): they contain the case
executables in `/opt/nek5000/cases/<case>/nek5000`, and the `run_*_docker.sh` scripts put the right one on the PATH.

```bash
# Check NEK5000 is available (in the images, e.g. for TCFmini_3D_Re180)
export PATH=/opt/nek5000/cases/mini_channel:$PATH
which nek5000
```

The examples start 1 Python and 10 (or more) Nek5000 ranks. Open MPI 5 counts slots by physical core: on a machine
with fewer cores than ranks, pass `--oversubscribe` to `mpirun` (the scripts take extra `mpirun` options from
`MPIRUN_ARGS`, e.g. `MPIRUN_ARGS=--oversubscribe ./run_nekenv_docker.sh`).

---

**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
