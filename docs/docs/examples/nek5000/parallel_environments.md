---
sidebar_position: 3
---

# Parallel Multi-Agent Interface

[`NekParallelEnv`](../../api/nek/parallel_env) wraps a base [`NekEnv`](../../api/nek/env) and exposes a dictionary-based multi-agent API in which each actuator is treated as an independent agent. Every agent receives its own observation array, produces its own scalar action, and can be assigned a per-agent reward signal.

All scripts live in [`examples/nek/getting_started/2_parallel_env/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/2_parallel_env).

## Creating an environment

```python
from hydrogym.nek import NekEnv, NekParallelEnv

env_config = {'environment_name': 'TCFmini_3D_Re180', 'nproc': 10}
base_env = NekEnv(env_config=env_config)

env = NekParallelEnv(base_env)

observations, infos = env.reset()
# observations: {'agent_0': array([...]), 'agent_1': array([...]), ...}

actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

Each `'agent_N'` key corresponds to one actuator degree of freedom in the underlying NEK5000 case.

## Example scripts

| File | Purpose |
|------|---------|
| [`test_nek_parallel.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/2_parallel_env/test_nek_parallel.py) | Runs the dict-based multi-agent interface with random actions, printing per-agent observations and rewards |
| [`train_sb3_parallel.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/2_parallel_env/train_sb3_parallel.py) | Demonstrates a DIY centralised wrapper that concatenates all agents into a single array, enabling direct SB3 training |
| [`run_parallel_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/2_parallel_env/run_parallel_docker.sh) | Docker/MPI launcher |

## Running the examples

```bash
# Test the parallel environment
mpirun -np 1 python test_nek_parallel.py --steps 100 : -np 10 nek5000

# Train with DIY centralised wrapper
mpirun -np 1 python train_sb3_parallel.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## SB3 integration

`NekParallelEnv` is **not** directly compatible with SB3 because SB3 expects a flat NumPy array for observations, not a dictionary. Two approaches are shown in the examples:

**DIY centralised wrapper** (shown in [`train_sb3_parallel.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/2_parallel_env/train_sb3_parallel.py)) — manually concatenates all agent observations into one array and splits the policy output back into per-agent actions. This is an educational approach that makes the data flow explicit, at the cost of ~50 lines of wrapper code.

**SuperSuit** (shown in the [PettingZoo](./pettingzoo) page) — the production-ready approach. SuperSuit provides a single `pettingzoo_env_to_vec_env_v1` call that handles observation padding, action-space normalisation, and agent-termination logic, converting the multi-agent environment into an SB3-compatible vectorised environment in a few lines.

If you are building a new training pipeline, the [PettingZoo + SuperSuit](./pettingzoo) approach is recommended. The DIY wrapper is most useful when you need fine-grained control over how agent data is aggregated, or when learning how the conversion works.

## When to use this interface

- Multiple actuators that each need an independent observation and reward
- True MARL scenarios where per-agent credit assignment is important
- Frameworks (RLlib, MARLlib) that natively consume PettingZoo parallel environments
- Inspecting per-agent reward distributions during debugging

For a single centralised policy that handles all actuators, the [Single Agent](./single_environment) interface is simpler.
