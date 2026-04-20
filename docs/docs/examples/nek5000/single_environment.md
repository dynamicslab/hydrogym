---
sidebar_position: 2
---

# Single Agent Interface

In this example we will demonstrate how to have a single agent interact with HydroGym's gymnasium-derived interface for a single-agent Nek5000 environment.

## Interface

Following the standard HydroGym interface, see [Firedrake's Getting Started](../firedrake/getting_started.md), we define the Nek environment in the usual manner:

```python
env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nproc': 10,
    'use_clean_cache': False,
    'configuration_file': None,  # Auto-detects config.yaml
}
env = NekEnv(env_config=env_config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## Files

In the case of this example, we have 3 main files to aid us in running the single agent case:

- [`test_nek_direct.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/test_nek_direct.py): Basic environment test with zero control
- [`train_sb3_nek_direct.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/train_sb3_nek_direct.py): SB3 training (PPO/TD3/SAC) with Monitor & VecNormalize
- [`run_nekenv_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/1_nekenv_single/run_nekenv_docker.sh): Docker/MPI execution script

## Usage

### Testing the Environment

To test the environment, we have `test_nek_direkt`, which we can launch with MPI

```bash
mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000
```

It takes the following arguments:

- `steps`: the number of steps to run
- `env`: the name of the environment
- `nproc`: the number of Nek5000 processes
- `local-dir`: local fallback directory for the environment
- `config-file`: the config file to be used

### Train the Reinforcement Learning Agent

To then train a single reinforcement learning agent, we utilize `train_sb3_nek_direct`. E.g.:

```bash
mpirun -np 1 python train_sb3_nek_direct.py --env MiniChannel_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

It takes the following input arguments:

- `env`: the name of the environment
- `local-dir`: the local environment directory
- `nproc`: the number of Nek5000 processes
- `work-dir`: the work directory
- `config-file`: config file to be used

In addition there are a number of algorithm-specific arguments, which can be set and played with:

- `algo`: the reinforcement learning algorithm, choices here are PPO, TD3, or SAC
- `total-timesteps`: number of total timesteps, $100,000$ is set as the default
- `n-steps`: in the case of PPO, the number of steps per update
- `learning-rate`: the learning rate
- `batch-size`: batch size of the algorithm, the default is $64$
- `gamma`: default value of $0.99$

For logging and storing checkpoints, we then have

- `log-dir`: the logging directory
- `save-freq`: the frequency at which results are stored

## When to Use

- Single actuator/sensor scenarios
- Direct SB3 compatibility (Monitor, VecNormalize wrappers included)
- Simple baseline comparisons with zero control

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
