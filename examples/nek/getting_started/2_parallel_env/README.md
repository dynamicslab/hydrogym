# NEK5000 Parallel Multi-Agent Interface (parallel_env)

Dict-based multi-agent interface where each actuator is treated as a separate agent.

## Interface

```python
from hydrogym.nek import NekEnv, NekParallelEnv

# Create base environment
env_config = {'environment_name': 'TCFmini_3D_Re180', 'nproc': 10}
base_env = NekEnv(env_config=env_config)

# Wrap with parallel interface
env = NekParallelEnv(base_env)

# Dict-based API
observations, infos = env.reset()
# observations: {'agent_0': array, 'agent_1': array, ...}

actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

## Files

- `test_nek_parallel.py` - Test parallel environment with dict-based interface
- `train_sb3_parallel.py` - SB3 training with **DIY centralized wrapper** (educational)
- `run_parallel_docker.sh` - Docker/MPI execution script

## Usage

### Test Environment
```bash
mpirun -np 1 python test_nek_parallel.py --steps 100 : -np 10 nek5000
```

### Train RL Agent (DIY Centralized Approach)
```bash
mpirun -np 1 python train_sb3_parallel.py --env MiniChannel_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## When to Use

- Multiple agents with independent observation/action spaces
- Dict-based multi-agent interface needed
- True MARL scenarios (with RLlib or custom frameworks)
- Per-agent reward inspection

## SB3 Integration

**parallel_env is NOT directly compatible with SB3** (SB3 expects arrays, not dicts).

**Solutions:**

1. **DIY Centralized Wrapper** (Chapter 2 - Educational)
   - Shown in `train_sb3_parallel.py`
   - Manually concatenates all agents → single array
   - Educational: shows how conversion works

2. **SuperSuit** (Chapter 3 - Production)
   - See chapter 3 for PettingZoo + SuperSuit approach
   - Production-ready ecosystem solution

## Notes

- Each agent controls one actuator (scalar action)
- Each agent receives local observations
- Rewards can be per-agent or shared
- For simple centralized control, just use `NekEnv` directly (Chapter 1)
