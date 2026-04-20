---
sidebar_position: 1
---

# Getting Started

**START HERE** for NEK5000-based RL interface examples using `env.reset()` and `env.step()`.

This directory contains comprehensive examples for using HydroGym's NEK5000-based flow environments with different interface patterns, from single-agent to multi-agent reinforcement learning.

> **Note:** NEK5000 requires MPI for parallel execution. All examples use `mpirun` to coordinate between the Python controller and NEK5000 solver processes.

## Directory Structure

Each subdirectory demonstrates a specific interface pattern with complete examples:

### 1. [`1_nekenv_single/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/1_nekenv_single) - Single Agent (Standard Gym)

**Interface:** `NekEnv` - Standard Gymnasium single-agent interface
**Use Case:** Single actuator/sensor scenarios
**SB3 Compatible:** ✅ Direct (no wrapper needed)

```python
from hydrogym.nek import NekEnv

env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nproc': 10,
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

- `test_nek_direct.py` - Basic environment test with zero control
- `train_sb3_nek_direct.py` - SB3 training with Monitor & VecNormalize
- `run_nekenv_docker.sh` - Docker/MPI execution script

---

### 2. [`2_parallel_env/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/2_parallel_env) - Multi-Agent Parallel (PettingZoo)

**Interface:** `parallel_env` - PettingZoo parallel multi-agent
**Use Case:** Multiple independent agents with simultaneous actions
**SB3 Compatible:** ⚠️ Requires wrapper (SuperSuit or custom)

```python
from hydrogym.nek import parallel_env

env = parallel_env(
    environment_name='TCFmini_3D_Re180',
    nproc=10,
    num_agents=3,  # Multiple agents
)

# Dictionary-based observations and actions
obs = env.reset()  # {'agent_0': array, 'agent_1': array, 'agent_2': array}
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

**Files:**

- `test_nek_parallel.py` - Multi-agent environment test
- `train_sb3_parallel.py` - SB3 training with SuperSuit wrappers
- `run_parallel_docker.sh` - Docker/MPI execution script

---

### 3. [`3_pettingzoo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/3_pettingzoo) - PettingZoo AEC Interface

**Interface:** PettingZoo AEC (Agent Environment Cycle)
**Use Case:** Turn-based multi-agent scenarios
**Configuration file:** YAML configs are used to lock simulation and runner settings for reproducible training. 
**SB3 Compatible:** ⚠️ Requires wrapper

```python
from hydrogym.nek import parallel_env
from pettingzoo.utils import parallel_to_aec

parallel = parallel_env(environment_name='TCFmini_3D_Re180', nproc=10)
env = parallel_to_aec(parallel)

# Turn-based API
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = env.action_space(agent).sample()
    env.step(action)
```

**Files:**

- `test_nek_pettingzoo.py` - AEC interface test
- `train_sb3_pettingzoo.py` - Training with turn-based agents
- `run_pettingzoo_docker.sh` - Docker/MPI execution script

---

### 4. [`4_from_hf/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/4_from_hf) - HuggingFace Data Manager

**Interface:** Load pre-packaged environments from HuggingFace Hub or local directories
**Use Case:** Using standardized, version-controlled environment configurations
**SB3 Compatible:** ✅ Works with any env type

```python
from hydrogym.nek import NekDataManager, NekEnv

# Initialize data manager
dm = NekDataManager(local_dir="./packaged_envs")

# Prepare workspace (downloads/extracts if needed)
config = dm.prepare_workspace(
    env_name="TCFmini_3D_Re180",
    nproc=10,
)

# Create environment
env = NekEnv(env_config=config)
```

**Files:**

- `test_nek_DM.py` - Data manager test
- `train_sb3_from_hf.py` - Training with HF environments
- `run_from_hf_docker.sh` - Docker/MPI execution script

---

### 5. [`5_hydrogym_control/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/5_hydrogym_control) - HydroGym Controllers + Integrate

**Interface:** Using `hgym.integrate()` for time-stepping with controllers
**Use Case:** Classical control, RL deployment, or hybrid control strategies
**SB3 Compatible:** ✅ Pass trained model as controller

```python
from hydrogym import integrate
from hydrogym.nek import NekEnv

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
- `run_control_docker.sh` - Docker/MPI execution script

---

### 6. [`6_zeroshot_wing_demo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/6_zeroshot_wing_demo) - Zero-Shot Wing Deployment

**Interface:** `NekEnv` + PettingZoo rollout with deployment controllers
**Use Case:** Deploy pre-trained/legacy DRL policies on small wing without new training
**SB3 Compatible:** ✅ For loading trained policies; demo script is rollout-only

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)
```

**Files:**

- `test_nek_pettingzoo.py` - zero-shot wing rollout demo
- `meta_policy_small_wing_template.py` - template for explicit legacy `MetaPolicy.py` usage
- `run_pettingzoo_docker.sh` - Docker/MPI execution script

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
mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000
```

### 3. Train an RL Agent

```bash
mpirun -np 1 python train_sb3_nek_direct.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## Comparison Table

| Directory | Interface | Obs Format | Action Format | SB3 Direct | Best For |
| --------- | --------- | ---------- | ------------- | ---------- | -------- |
| **1_nekenv_single** | `NekEnv` | Array | Array | ✅ Yes | Single actuator, simple baseline |
| **2_parallel_env** | `parallel_env` | Dict | Dict | ⚠️ Wrapper | Independent multi-agent scenarios |
| **3_pettingzoo** | AEC | Sequential | Sequential | ⚠️ Wrapper | Turn-based agents |
| **4_from_hf** | Any | Depends | Depends | Depends | Reproducible, versioned environments |
| **5_hydrogym_control** | Any + `integrate()` | Any | Any | ✅ Yes | Classical + RL hybrid control |
| **6_zeroshot_wing_demo** | PettingZoo Parallel | Dict | Dict | ✅ Deployment | Small-wing zero-shot DRL rollout |

## Requirements

### NEK5000 Setup

NEK5000 must be compiled and the `nek5000` executable must be in your PATH or specified in the environment configuration. We highly recommend using the provided Docker container.

```bash
# Check NEK5000 is available
which nek5000

# Or set path explicitly in config
env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nek_path': '/path/to/nek5000',
    'nproc': 10,
}
```

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
