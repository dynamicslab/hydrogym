---
sidebar_position: 4
---

# The PettingZoo Interface

To enable multi-agent interactions, HydroGym enables the interfacing with the Farama Foundation's [PettingZoo standard](https://pettingzoo.farama.org/) for multi-agent reinforcement learning. Training itself can then be performed with the production-ready SB3 via [SuperSuit](https://pettingzoo.farama.org/api/wrappers/supersuit_wrappers/).

## Interface

To set up the multi-agent environment, we begin by first constructing the environment in the usual manner, before converting it into a multi-agent environment with the `make_petting_zoo` wrapper:

```python
from hydrogym.nek import NekEnv, make_pettingzoo_env

# Create base environment
env_config = {'environment_name': 'TCFmini_3D_Re180', 'nproc': 10}
base_env = NekEnv(env_config=env_config)

# Wrap with PettingZoo interface
env = make_pettingzoo_env(base_env)
```

At which point we can then utilize PettingZoo's parallel API to interact with the environment

```python
observations, infos = env.reset()
actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

## Files

Just like with the other examples, 3 main files are provided with the example:

- [`test_nek_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/test_nek_pettingzoo.py): Test PettingZoo interface
- [`train_sb3_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/train_sb3_pettingzoo.py): **SB3 training with SuperSuit** (production approach)
- [`run_pettingzoo_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/run_pettingzoo_docker.sh): Docker/MPI execution

## Usage

### Testing the Environment

To test the environment, and its interaction with the Nek solver we utilize the `test_nek_pettingzoo` script:

```bash
mpirun -np 1 python test_nek_pettingzoo.py --steps 100 : -np 10 nek5000
```

### Train RL Agent (SuperSuit Production Approach)

After whuch we can then train the reinforcement learning agent following the recommended approach by [SuperSuit](https://pettingzoo.farama.org/api/wrappers/supersuit_wrappers/). Here we will utilize the pettingzoo training helper script, which we can configure on the command line:

```bash
mpirun -np 1 python train_sb3_pettingzoo.py --env MiniChannel_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## Configuration-Driven Tutorial (Recommended for Reproducibility)

While the command line is highly useful for quick iterations in testing, once we move to structured experimental studies, we want to be able to rerun experiments more quickly, and in a scripted fashion. For this purpose, we provide the option to utilize a YAML configuration file to lock the simulation, and runner settings across runs. To utilize the YAML file-based approach we first have to begin by preparing the workspace:

```bash
python ../prepare_workspace.py \
  --local-dir ../../../packaged_envs \
  --env TCFmini_3D_Re180 \
  --work-dir ./train_run
```

Before we can then train with a configuration file

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

And can then perform rollouts for evaluation

```bash
cd train_run
mpirun -np 1 python ../test_nek_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../configs/pettingzoo_tcfmini_re180.yml \
  --steps 2500 \
  : -np 10 nek5000
```

**Notes:**

> - The config lives in `examples/nek/configs/pettingzoo_tcfmini_re180.yml`.
> - Run from the workspace (`train_run`) so `compile_path: "."` resolves to case files.
> - Ensure `--nproc` matches `simulation.nproc` in the config.

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

**Notes:**

> - SuperSuit is the **recommended** approach for production
> - PettingZoo API ensures compatibility with many MARL tools
