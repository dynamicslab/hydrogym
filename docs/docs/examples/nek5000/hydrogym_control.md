---
sidebar_position: 6
---

# Time-Stepping with `integrate()`

The [`integrate()`](../../api/nek/integrate) function provides a convenient time-stepping loop that drives a HydroGym environment with an arbitrary controller. The controller can be a trained SB3 model, a classical feedback law, or any Python callable that maps `(t, obs, env)` to an action array. This makes `integrate()` the natural tool for evaluating trained policies on longer rollouts and for comparing RL and classical control strategies under identical conditions.

All scripts live in [`examples/nek/getting_started/5_hydrogym_control/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/5_hydrogym_control).

## Basic usage

```python
from hydrogym.nek import NekEnv, integrate

env = NekEnv.from_hf('TCFmini_3D_Re80', nproc=10)

# Classical controller: opposition control
def opposition_control(t, obs, env):
    return -obs[:env.action_space.shape[0]]

integrate(env, controller=opposition_control, num_steps=1000)

# Or pass a trained SB3 model directly
from stable_baselines3 import PPO
model = PPO.load("model.zip")
integrate(env, controller=model, num_steps=1000)
```

`integrate()` handles the `reset()` → `step()` loop internally. For SB3 models it automatically calls `model.predict()` with the current observation; for callables it passes `(t, obs, env)`.

## Example scripts

| File | Purpose |
|------|---------|
| [`test_nek_env_controller.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/5_hydrogym_control/test_nek_env_controller.py) | Runs several classical controllers in sequence and prints the resulting rewards |
| [`train_sb3_with_integrate.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/5_hydrogym_control/train_sb3_with_integrate.py) | Complete train → save → load → evaluate workflow, including a comparison between the trained policy and a classical baseline |
| [`run_control_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/5_hydrogym_control/run_control_docker.sh) | Docker/MPI launcher |

## Running the examples

```bash
# Test controllers
mpirun -np 1 python test_nek_env_controller.py \
    --config test_config.yml \
    : -np 10 nek5000

# Full train + evaluate workflow
mpirun -np 1 python train_sb3_with_integrate.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 50000 \
    --eval-steps 200 \
    : -np 10 nek5000
```

## Supported controller types

### Classical controllers

Any Python callable with the signature `action = controller(t, obs, env)`:

```python
import numpy as np

def opposition_control(t, obs, env):
    return -obs[:env.action_space.shape[0]] * 0.5

def sinusoidal(t, obs, env):
    return np.array([np.sin(2 * np.pi * t)], dtype=np.float32)

def zero_control(t, obs, env):
    return np.zeros(env.action_space.shape, dtype=np.float32)
```

### RL controllers

Any SB3 model with a `.predict()` method:

```python
from stable_baselines3 import PPO
model = PPO.load("trained_model.zip", env=env)
integrate(env, controller=model, num_steps=1000)
```

:::note
If the policy was trained with `VecNormalize`, wrap the evaluation environment with the same normalisation statistics before calling `integrate()`:

```python
env_eval = VecNormalize.load("vec_normalize.pkl", DummyVecEnv([lambda: base_env]))
model = PPO.load("model.zip")
integrate(env_eval, controller=model, num_steps=1000)
```
:::

## Complete train + evaluate workflow

The `train_sb3_with_integrate.py` script demonstrates the full lifecycle:

```python
from hydrogym.nek import NekEnv, integrate
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Phase 1: training ---
env = Monitor(NekEnv(env_config=config))
env = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50_000)
model.save("model.zip")
env.save("vec_normalize.pkl")

# --- Phase 2: evaluation ---
base_eval = NekEnv(env_config=config)
env_eval = VecNormalize.load("vec_normalize.pkl", DummyVecEnv([lambda: base_eval]))
model = PPO.load("model.zip")

integrate(env_eval, controller=model, num_steps=1000)
```

## When to use `integrate()`

- Evaluating a trained policy on a rollout that is longer than the training episode length
- Comparing multiple controllers under identical initial conditions
- Running a classical control law without the overhead of the full `env.step()` bookkeeping
- Generating time-series data for analysis or visualisation

For interactive training loops where you need fine-grained control over the step-by-step interaction, use `env.reset()` and `env.step()` directly.
