# NEK5000 Control with integrate()

Use `integrate()` for time-stepping simulation with classical or RL controllers.

## Interface

```python
from hydrogym.nek import NekEnv, integrate

# Create environment
env = NekEnv.from_hf('TCFmini_3D_Re80', nproc=10)

# Classical controller
def opposition_control(t, obs, env):
    return -obs[:env.action_space.shape[0]]

integrate(env, controller=opposition_control, num_steps=1000)

# Or use trained SB3 model
from stable_baselines3 import PPO
model = PPO.load("model.zip")
integrate(env, controller=model, num_steps=1000)
```

## Files

- `test_nek_env_controller.py` - Test various controllers with integrate()
- `train_sb3_with_integrate.py` - **Complete workflow: train + evaluate + compare**
- `run_control_docker.sh` - Docker/MPI execution

## Usage

### Test Controllers
```bash
mpirun -np 1 python test_nek_env_controller.py --config test_config.yml : -np 10 nek5000
```

### Train & Evaluate Workflow
```bash
mpirun -np 1 python train_sb3_with_integrate.py --env MiniChannel_Re180 --algo PPO --total-timesteps 50000 --eval-steps 200 : -np 10 nek5000
```

## When to Use

- **Evaluating trained RL policies** on longer rollouts
- Comparing RL vs classical control strategies
- Time-stepping simulations with custom control laws
- Benchmarking different control approaches

## Controllers Supported

### Classical Controllers
Custom functions with signature: `action = controller(t, obs, env)`

- **Opposition Control**: `action = -alpha * observation`
- **Blowing/Suction**: `action = constant`
- **Sinusoidal**: `action = sin(omega * t)`
- **Zero Control**: `action = 0` (baseline)

```python
def opposition_control(t, obs, env):
    return -obs[:env.action_space.shape[0]] * 0.5

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
env = NekEnv(env_config=config)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, ...)

model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=50000)
model.save("model.zip")
env.save("vec_normalize.pkl")

# Phase 2: Evaluation
env_eval = NekEnv(env_config=config)
env_eval = DummyVecEnv([lambda: env_eval])
env_eval = VecNormalize.load("vec_normalize.pkl", env_eval)
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
- Compatible with any environment type (NekEnv, parallel_env, etc.)
- RL policies need the same normalization wrapper used during training
- Classical controllers don't need normalization wrappers
- Use for evaluation and comparison, not for training
