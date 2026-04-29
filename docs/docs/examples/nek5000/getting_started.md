---
sidebar_position: 1
---

# Getting Started with NEK5000

This section covers HydroGym's NEK5000-based environments, from a simple single-agent setup to multi-agent training and zero-shot policy deployment. Each sub-page corresponds to a numbered subdirectory in [`examples/nek/getting_started/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started) and focuses on a distinct interface pattern.

:::note
NEK5000 environments use MPMD (Multiple Program Multiple Data) execution: a Python controller process and one or more NEK5000 solver processes are launched together via `mpirun`. The colon syntax (`: -np N nek5000`) is specific to this launch pattern and is required for all NEK5000 examples.
:::

## Interface patterns at a glance

| Sub-page | Directory | Interface | SB3 direct? | Best for |
|----------|-----------|-----------|-------------|----------|
| [Single agent](./single_environment) | [`1_nekenv_single/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/1_nekenv_single) | [`NekEnv`](../../api/nek/env) | ✅ Yes | Single actuator, simple baselines |
| [Parallel multi-agent](./parallel_environments) | [`2_parallel_env/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/2_parallel_env) | [`NekParallelEnv`](../../api/nek/parallel_env) | ⚠️ Needs wrapper | Independent multi-agent |
| [PettingZoo](./pettingzoo) | [`3_pettingzoo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/3_pettingzoo) | PettingZoo + SuperSuit | ✅ Via SuperSuit | Production multi-agent training |
| [HuggingFace Hub](./from_hf) | [`4_from_hf/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/4_from_hf) | `NekEnv.from_hf()` | ✅ Any type | Minimal configuration, recommended default |
| [integrate()](./hydrogym_control) | [`5_hydrogym_control/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/5_hydrogym_control) | [`integrate()`](../../api/nek/integrate) | ✅ Pass model | Classical or RL evaluation |
| [Zero-shot wing](./zeroshot_wing) | [`6_zeroshot_wing_demo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/6_zeroshot_wing_demo) | PettingZoo + MetaPolicy | ✅ Deployment only | Pre-trained multi-policy deployment |

## Choosing an interface

- **New to HydroGym or training a first agent?** Start with the [Single Agent](./single_environment) interface (`NekEnv`). It is directly compatible with Stable-Baselines3 and requires no wrapper code.
- **Multiple independent actuators, each with its own observation?** The [Parallel Multi-Agent](./parallel_environments) interface (`NekParallelEnv`) gives each actuator its own observation and action space in a dictionary-based API.
- **Production multi-agent training with SB3?** The [PettingZoo](./pettingzoo) page shows how to use SuperSuit to bridge the PettingZoo API to SB3's vectorised environment interface.
- **Want the least configuration overhead?** Use `NekEnv.from_hf()` as shown on the [HuggingFace Hub](./from_hf) page — it auto-detects configuration files and handles workspace setup.
- **Evaluating a trained policy or comparing controllers?** The [`integrate()`](./hydrogym_control) helper runs a time-stepping loop that accepts either an SB3 model or a Python callable as the controller.
- **Deploying multiple pre-trained policies on a wing?** See the [Zero-Shot Wing](./zeroshot_wing) demo.

## Quick start

```bash
# Test the single-agent environment (10 MPI ranks: 1 controller + 10 solver)
cd 1_nekenv_single/
mpirun -np 1 python test_nek_direct.py --steps 100 : -np 10 nek5000

# Train a PPO agent
mpirun -np 1 python train_sb3_nek_direct.py \
    --env TCFmini_3D_Re180 \
    --algo PPO \
    --total-timesteps 100000 \
    : -np 10 nek5000
```

## NEK5000 setup

NEK5000 must be compiled and available as `nek5000` on your `PATH`, or its location must be given explicitly in the environment configuration. The provided Docker image includes a pre-compiled binary and is the recommended way to run these examples.

```bash
# Verify that NEK5000 is available
which nek5000

# Or specify the path in the environment config
env_config = {
    'environment_name': 'TCFmini_3D_Re180',
    'nek_path': '/opt/nek5000/bin/nek5000',
    'nproc': 10,
}
```
