# NEK5000 Single Agent Interface (NekEnv)

Standard Gym interface for single-agent NEK5000 environments with direct instantiation.

## Interface

```python
env_config = {
    "environment_name": "TCFmini_3D_Re180",
    "nproc": 10,
    "use_clean_cache": False,
    "configuration_file": None,  # Auto-detects config.yaml
}
env = NekEnv(env_config=env_config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Files

- `test_nek_direct.py` - Basic environment test with opposition control
- `train_sb3_nek_direct.py` - SB3 training (PPO/TD3/SAC) with Monitor & VecNormalize
- `run_nekenv_docker.sh` - MPMD launch script (runs as is in the HydroGym images)

## Usage

### Test Environment
```bash
# from a prepared workspace: the Nek5000 case files must be in the working directory
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ws && cd ws
mpirun -np 1 python ../test_nek_direct.py --steps 100 : -np 10 nek5000
```

### Train RL Agent
```bash
# in the workspace ws/ of the block above
mpirun -np 1 python ../train_sb3_nek_direct.py --env TCFmini_3D_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## When to Use

- Single actuator/sensor scenarios
- Direct SB3 compatibility (Monitor, VecNormalize wrappers included)
- Simple baseline comparisons with opposition control
