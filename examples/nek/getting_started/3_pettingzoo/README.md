# NEK5000 PettingZoo Interface

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
