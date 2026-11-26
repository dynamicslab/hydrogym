# NEK5000 Control with integrate()

Use `integrate()` for time-stepping simulation with classical or RL controllers.

## Interface

```python
from hydrogym.nek import NekEnv, integrate

# Create environment
env = NekEnv.from_hf("TCFmini_3D_Re180", nproc=10)


# Classical controller
def opposition_control(t, obs, env):
    return -obs[1::2]  # obs is [u', v'] per actuator: oppose the wall-normal velocity


integrate(env, controller=opposition_control, num_steps=1000)

# Or use trained SB3 model
from stable_baselines3 import PPO

model = PPO.load("model.zip")
integrate(env, controller=model, num_steps=1000)
```

## Files

- `test_nek_env_controller.py` - Test various controllers with integrate()
- `train_sb3_with_integrate.py` - **Complete workflow: train + evaluate + compare**
- `run_control_docker.sh` - MPMD launch script (runs as is in the HydroGym images)

## Usage

### Test Controllers
```bash
# from a prepared workspace: the Nek5000 case files must be in the working directory
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ws && cd ws
mpirun -np 1 python ../test_nek_env_controller.py --env TCFmini_3D_Re180 --nproc 10 : -np 10 nek5000   # or --config <your_config>.yml --nproc 10
```

### Train & Evaluate Workflow
```bash
mpirun -np 1 python ../train_sb3_with_integrate.py --env TCFmini_3D_Re180 --nproc 10 --algo PPO --total-timesteps 50000 --eval-steps 200 : -np 10 nek5000
```

## When to Use

- **Evaluating trained RL policies** on longer rollouts
- Comparing RL vs classical control strategies
- Time-stepping simulations with custom control laws
- Benchmarking different control approaches

## Controllers Supported

### Classical Controllers
Custom functions with signature: `action = controller(t, obs, env)`

- **Opposition Control**: `action = -alpha * v'`, the wall-normal component of each actuator's observation (`obs[1::2]`)
- **Blowing/Suction**: `action = constant`
- **Sinusoidal**: `action = sin(omega * t)`
- **Zero Control**: `action = 0` (baseline)

```python
def opposition_control(t, obs, env):
    return -obs[1::2] * 0.5


def zero_control(t, obs, env):
    return np.zeros(env.action_space.shape, dtype=np.float32)
```

### RL Controllers
Any SB3 model with `.predict()` method:

```python
from stable_baselines3 import PPO

model = PPO.load("trained_model.zip", env=env)
integrate(env, controller=model, num_steps=1000)
```

## Complete Train + Evaluate Workflow

The `train_sb3_with_integrate.py` script demonstrates the full workflow:

1. **Train RL agent** with SB3 (Monitor, VecNormalize)
2. **Save model and normalization stats**
3. **Load trained model** for evaluation
4. **Evaluate with integrate()** for extended rollouts
5. **Compare RL vs classical controllers**

```python
# Phase 1: Training
base_env = NekEnv(env_config=config)
env = DummyVecEnv([lambda: Monitor(base_env)])
env = VecNormalize(env, ...)

model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=50000)
model.save("model.zip")
env.save("vec_normalize.pkl")

# Phase 2: Evaluation, in the same environment: one MPMD launch serves one NekEnv (a second NekEnv(...)
# would wait for Nek5000 ranks that are already connected)
env_eval = DummyVecEnv([lambda: base_env])
env_eval = VecNormalize.load("vec_normalize.pkl", env_eval)
env_eval.training, env_eval.norm_reward = False, False
model = PPO.load("model.zip")

integrate(env_eval, controller=model, num_steps=1000)
```

## Key Features

- **Controller agnostic**: Works with RL models and classical functions
- **Automatic handling**: Detects controller type automatically
- **Normalization support**: Properly handles VecNormalize for RL policies
- **Comparison tool**: Evaluate multiple controllers in sequence

## Notes

- `integrate()` handles the time-stepping loop internally
- Compatible with any environment type (NekEnv, parallel_env, etc.); one MPMD launch serves one environment, so
  evaluate in the environment that was trained
- RL policies need the same normalization wrapper used during training
- Classical controllers don't need normalization wrappers
- Use for evaluation and comparison, not for training
