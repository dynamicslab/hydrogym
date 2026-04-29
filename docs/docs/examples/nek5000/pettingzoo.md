---
sidebar_position: 4
---

# PettingZoo Interface

The [PettingZoo](https://pettingzoo.farama.org/) wrapper provides a standardised multi-agent API and, through [SuperSuit](https://github.com/Farama-Foundation/SuperSuit), a production-ready path to Stable-Baselines3 training on multi-agent NEK5000 environments.

All scripts live in [`examples/nek/getting_started/3_pettingzoo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/3_pettingzoo).

## Creating an environment

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

env_config = {'environment_name': 'TCFmini_3D_Re180', 'nproc': 10}
base_env = NekEnv(env_config=env_config)

env = make_pettingzoo_env(base_env)

observations, infos = env.reset()
actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

[`make_pettingzoo_env`](../../api/nek/pettingzoo_env) wraps any `NekEnv` instance and returns a PettingZoo [Parallel API](https://pettingzoo.farama.org/api/parallel/) environment.

## Example scripts

| File | Purpose |
|------|---------|
| [`test_nek_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/test_nek_pettingzoo.py) | Runs the PettingZoo interface with random actions and verifies the observation/action shapes |
| [`train_sb3_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/train_sb3_pettingzoo.py) | Production SB3 training via SuperSuit |
| [`run_pettingzoo_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/run_pettingzoo_docker.sh) | Docker/MPI launcher |

## Running the examples

```bash
# Test
mpirun -np 1 python test_nek_pettingzoo.py --steps 100 : -np 10 nek5000

# Train with SuperSuit + SB3
mpirun -np 1 python train_sb3_pettingzoo.py \
    --env TCFmini_3D_Re180 \
    --nproc 10 \
    --algo TD3 \
    --total-timesteps 5000000 \
    : -np 10 nek5000
```

## Configuration-driven workflow

For reproducible training runs, use a YAML configuration file to lock all simulation and runner settings:

```bash
# 1. Prepare a workspace
python ../prepare_workspace.py \
  --local-dir ../../../packaged_envs \
  --env TCFmini_3D_Re180 \
  --work-dir ./train_run

# 2. Train from a config file (run from the workspace directory)
cd train_run
mpirun -np 1 python ../train_sb3_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../configs/pettingzoo_tcfmini_re180.yml \
  --algo TD3 \
  --total-timesteps 5000000 \
  : -np 10 nek5000

# 3. Evaluate
mpirun -np 1 python ../test_nek_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../configs/pettingzoo_tcfmini_re180.yml \
  --steps 2500 \
  : -np 10 nek5000
```

The YAML config lives at `examples/nek/configs/pettingzoo_tcfmini_re180.yml`. Run training from the workspace directory (`train_run/`) so that the `compile_path: "."` in the config resolves to the prepared case files. Make sure `--nproc` matches `simulation.nproc` in the config.

## SB3 integration with SuperSuit

SuperSuit is PettingZoo's official wrapper library. It handles observation-space alignment, agent termination, and the conversion to a vectorised Gym-compatible environment — all required steps before passing a multi-agent environment to SB3.

```bash
pip install pettingzoo supersuit
```

The training script applies four SuperSuit wrappers in sequence:

1. `pad_observations_v0` — pads observations to a uniform size across agents
2. `pad_action_space_v0` — pads action spaces to a uniform size
3. `black_death_v3` — handles early agent termination gracefully
4. `pettingzoo_env_to_vec_env_v1` — converts the PettingZoo parallel env to an SB3-compatible vectorised environment

## Comparison with the DIY wrapper

| | [DIY wrapper](./parallel_environments) | PettingZoo + SuperSuit |
|---|---|---|
| Dependencies | None extra | `pettingzoo`, `supersuit` |
| Lines of wrapper code | ~50 | ~5 |
| Purpose | Learning / debugging | Production training |
| Maintenance | Manual | Ecosystem-maintained |

## When to use this interface

- Production SB3 training on multi-agent environments
- Integration with the PettingZoo ecosystem (RLlib, MARLlib, etc.)
- Reproducible training with YAML configuration files
- Cases where SuperSuit's agent-lifecycle management is needed
