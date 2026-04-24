---
sidebar_position: 4
---

# PettingZoo Interface

PettingZoo-compatible wrapper for ecosystem integration and production-ready SB3 training via SuperSuit.

## Interface

```python
from hydrogym.nek import NekEnv, make_pettingzoo_env

# Create base environment
env_config = {'environment_name': 'TCFmini_3D_Re180', 'nproc': 10}
base_env = NekEnv(env_config=env_config)

# Wrap with PettingZoo interface
env = make_pettingzoo_env(base_env)

# PettingZoo parallel API
observations, infos = env.reset()
actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

## Files

- `test_nek_pettingzoo.py` - Test PettingZoo interface
- `train_sb3_pettingzoo.py` - **SB3 training with SuperSuit** (production approach)
- `run_pettingzoo_docker.sh` - Docker/MPI execution

## Usage

### Test Environment
```bash
mpirun -np 1 python test_nek_pettingzoo.py --steps 100 : -np 10 nek5000
```

### Train RL Agent (SuperSuit Production Approach)
```bash
mpirun -np 1 python train_sb3_pettingzoo.py --env MiniChannel_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## Configuration-Driven Tutorial (Recommended for Reproducibility)

Use a fixed YAML config to lock simulation + runner settings across runs.

### 1) Prepare a workspace
```bash
python ../prepare_workspace.py \
  --local-dir ../../../packaged_envs \
  --env TCFmini_3D_Re180 \
  --work-dir ./train_run
```

### 2) Train with a config file
```bash
cd train_run
mpirun -np 1 python ../train_sb3_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../configs/pettingzoo_tcfmini_re180.yml \
  --algo TD3 \
  --total-timesteps 5000000 \
  : -np 10 nek5000
```

### 3) Evaluate (PettingZoo rollouts)
```bash
cd train_run
mpirun -np 1 python ../test_nek_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../configs/pettingzoo_tcfmini_re180.yml \
  --steps 2500 \
  : -np 10 nek5000
```

Notes:
- The config lives in `examples/nek/configs/pettingzoo_tcfmini_re180.yml`.
- Run from the workspace (`train_run`) so `compile_path: "."` resolves to case files.
- Ensure `--nproc` matches `simulation.nproc` in the config.

## When to Use

- **Production SB3 training** on multi-agent environments
- PettingZoo ecosystem compatibility
- Using PettingZoo-specific tools and libraries
- Need standardized multi-agent API

## SB3 Integration with SuperSuit

**SuperSuit** is PettingZoo's official wrapper library for converting multi-agent envs to SB3-compatible format.

### Installation
```bash
pip install pettingzoo supersuit
```

### SuperSuit Wrappers Used
1. `pad_observations_v0` - Pad observations to uniform size
2. `pad_action_space_v0` - Pad action spaces to uniform size
3. `black_death_v3` - Handle agent termination
4. `pettingzoo_env_to_vec_env_v1` - Convert to vectorized Gym env

### Comparison with Chapter 2

| Aspect | Chapter 2 (DIY) | Chapter 3 (SuperSuit) |
|--------|-----------------|----------------------|
| Purpose | Educational | Production |
| Dependencies | None extra | pettingzoo, supersuit |
| Code complexity | ~50 lines wrapper | ~5 lines SuperSuit |
| Maintenance | DIY | Ecosystem-maintained |
| Use when | Learning | Deploying |

## Notes

- SuperSuit is the **recommended** approach for production
- Chapter 2 (DIY wrapper) is for understanding how it works
- PettingZoo API ensures compatibility with many MARL tools
