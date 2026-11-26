# NEK5000 environments from Hugging Face (from_hf)

Convenient method for loading environments with minimal configuration using `NekEnv.from_hf()`.

## Interface

```python
from hydrogym.nek import NekEnv

# Minimal configuration - just environment name and nproc
env = NekEnv.from_hf(
    "TCFmini_3D_Re180",
    nproc=10,
    use_clean_cache=False,
    local_fallback_dir=None,  # Optional: local environment directory
)

# That's it! Auto-detects config files, handles setup automatically
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Files

- `test_nek_DM.py` - Test from_hf() loading pattern
- `train_sb3_from_hf.py` - SB3 training with from_hf() pattern
- `run_from_hf_docker.sh` - MPMD launch script (runs as is in the HydroGym images)

## Usage

### Test Environment
```bash
# from a prepared workspace: the Nek5000 case files must be in the working directory
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ws && cd ws
mpirun -np 1 python ../test_nek_DM.py --steps 100 : -np 10 nek5000
```

### Train RL Agent
```bash
# in the workspace ws/ of the block above
mpirun -np 1 python ../train_sb3_from_hf.py --env TCFmini_3D_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## When to Use

- **Recommended for most users** - simplest approach
- Need minimal configuration (environment name + nproc)
- Want automatic config file detection
- Using standard/pre-packaged environments

## Comparison with Other Patterns

| Pattern | Configuration | Use Case |
|---------|--------------|----------|
| **from_hf()** (Chapter 4) | Minimal (name + nproc) | Recommended default |
| **Direct instantiation** (Chapter 1) | env_config dict | Full control needed |
| **Legacy config** | YAML + OmegaConf | Backwards compatibility |

## Benefits of from_hf()

- **Minimal code**: Just env name and nproc
- **Auto-detection**: Finds config files automatically
- **Fallback support**: Local directories or HuggingFace
- **Clean workspace**: No nested directories created
- **Caching**: Reuses prepared environments

## Notes

- Auto-detects `environment_config.yaml` or `config.yaml`
- `use_clean_cache=False` reuses existing prepared workspace
- `local_fallback_dir` allows using local environment packages, e.g. for HPC system usage
- For advanced configuration, use direct instantiation (Chapter 1)
