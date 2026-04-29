---
sidebar_position: 2
---

# Single Agent Interface

[`NekEnv`](../../api/nek/env) is HydroGym's standard single-agent interface for NEK5000 environments. It follows the [Gymnasium](https://gymnasium.farama.org/) API exactly, returning a flat NumPy array for observations and accepting a flat array for actions. This makes it directly compatible with Stable-Baselines3 without any additional wrapper code.

All scripts live in [`examples/nek/getting_started/1_nekenv_single/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/1_nekenv_single).

## Creating an environment

```python
from hydrogym.nek import NekEnv

env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nproc': 10,
    'use_clean_cache': False,  # reuse an existing prepared workspace
    'configuration_file': None,  # auto-detects environment_config.yaml or config.yaml
}

env = NekEnv(env_config=env_config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

The `nproc` parameter sets the number of NEK5000 solver processes. The total MPI rank count for the `mpirun` command is `1 + nproc` (one Python controller plus the solver ranks).

## Example scripts

| File | Purpose |
|------|---------|
| [`test_nek_direct.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/test_nek_direct.py) | Runs a fixed number of environment steps with zero control, printing observations and rewards |
| [`train_sb3_nek_direct.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/train_sb3_nek_direct.py) | SB3 training script with `Monitor` and `VecNormalize` wrappers; supports PPO, TD3, and SAC |
| [`run_nekenv_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/run_nekenv_docker.sh) | Convenience wrapper that handles Docker image selection and the `mpirun` launch |

## Running the examples

```bash
# Test the environment
mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000

# Train with PPO
mpirun -np 1 python train_sb3_nek_direct.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## Training with Stable-Baselines3

`NekEnv` is a drop-in Gymnasium environment, so it works with SB3's standard wrappers:

```python
from hydrogym.nek import NekEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nproc': 10,
}

env = NekEnv(env_config=env_config)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_tcfmini")
env.save("vec_normalize.pkl")
```

## When to use this interface

- Single actuator / sensor configurations
- Direct SB3 compatibility without wrapper code
- Establishing a simple baseline before moving to multi-agent setups
- Cases where a centralised policy receives all observations and produces all actions

For environments with multiple independent actuators that require separate observations and reward signals, see the [Parallel Multi-Agent](./parallel_environments) interface.
