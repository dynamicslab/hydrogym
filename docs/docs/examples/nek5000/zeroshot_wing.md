---
sidebar_position: 7
---

# Zero-Shot Wing Deployment (Multi-Policy MARL)

In this example, we will examine the scenario in which we seek to deploy a zero-shot controller for the flow over the canonical NACA4412 wing case. Multiple control policies are mapped to actuator subsets and executed together in one PettingZoo rollout.

> This is a deployment/evaluation demo only (no training). The template and controllers are intended for demonstration and should **not** be used to draw *physical conclusions*.

## Functionality of the Utility Script

We provide a utility script for working with functionalities of the Nek5000 backend which build upon the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) abstraction. This functionality is provided by [`test_nek_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/3_pettingzoo/test_nek_pettingzoo.py), which works as follows:

1. It loads a base `NekEnv` via `NekEnv.from_hf(...)` and wraps it with `make_pettingzoo_env(...)`
2. Constructs one controller per entry in `POLICY_SPECS` (from `meta_policy_small_wing_template.py`)
3. Assigns each controller to actuator agents by `x_range` and `side` (`SS` means `y > 0`, `PS` means `y < 0`)
4. Refreshes each group's actions every `drl_step` steps (refresh at step `0`; otherwise actions are held)
5. Clips actions to `action_bounds`
6. Computes an “inverted” + scaled reward summary for display (deployment-only)

Unassigned actuator agents receive zero actions meanwhile.

## Rollout Interface of PettingZoo

PettingZoo, in conjunction with HydroGym, then enables a rollout interface in line the wider reinforcement learning interfaces:

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)

obs_dict, info = env.reset()
actions = {agent: controller(obs_dict[agent]) for agent in env.agents}
obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
```

### Core Files

There are three main files which encapsulate the main functionality here:

- [`zetoshot_demo_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/zeroshot_demo_pettingzoo.py): Zero-shot multi-policy rollout demo (deployment only)
- [`meta_policy_small_wing_template.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/meta_policy_small_wing_template.py): Template defining `ENV_NAME`, `NPROC`, and `POLICY_SPECS`
- [`run_pettingzoo_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/run_pettingzoo_docker.sh): Runner script (module load + workspace prep + `mpirun`)

## Usage

### Recommended: Runner script

Utilizing the runner script, the syntax will look like the following:

```bash
./run_pettingzoo_docker.sh
./run_pettingzoo_docker.sh --policy-root /workspace/legacy_runs
```

### Run with Python Deployment Script

To start the computation with the Python-based deployment script, we will have to launch with MPI in the following manner:

```bash
mpirun -np 1 python test_nek_pettingzoo.py : -np 12 nek5000
```

If we have to use the legacy policy template, and are able to run as run root the syntax changes to the following:

```bash
mpirun -np 1 python test_nek_pettingzoo.py \
  --policy-template ./meta_policy_small_wing_template.py \
  --policy-root /path/to/legacy_runs \
  --steps 3000 \
  : -np 12 nek5000
```

The deployment script furthermore affords a number of useful overrides:

- `--policy-template PATH` (defaults to `./meta_policy_small_wing_template.py`)
- `--env ENV_NAME` (defaults from template `ENV_NAME`)
- `--nproc NPROC` (defaults from template `NPROC`)
- `--steps NUM_STEPS` (defaults from template `NUM_STEPS`)
- `--policy-root PATH` (where RL model run folders live)
- `--local-dir PATH` (optional fallback dir for packaged envs)
- `--log-every N` (reward table frequency)

## Policy Template

The template defines a lightweight legacy-`MetaPolicy.py`-style configuration. It requires the following top-level variables:

- `ENV_NAME`
- `NPROC`
- `NUM_STEPS`
- `POLICY_ROOT` (default for `--policy-root`)
- `POLICY_SPECS` (list of policy group dicts)

Each `POLICY_SPECS` entry supports:

- `name`
- `x_range: [x_min, x_max]`
- `side: "SS"` (y > 0) or `"PS"` (y < 0)
- `algorithm: "PPO" | "TD3" | "DDPG" | "BL" | "ZERO"`
- `drl_step` (action refresh interval; actions are held between refreshes)
- `action_bounds: [min, max]`
- optional scaling knobs: `u_tau`, `baseline_dudy`
- RL algorithms only: `agent_run_name`, `policy`, and/or `model_path`

Algorithm semantics:

- `ZERO` outputs an all-zero action (no model needed)
- `BL` outputs a constant action equal to `action_max` (no model needed)
- `PPO`/`TD3`/`DDPG` load a Stable-Baselines3 model from `model_path`/`POLICY_ROOT`

For overlapping actuator regions, the last-assigned policy takes precedence.

## Default RL Model Path Convention

For RL policies (`PPO`, `TD3`, `DDPG`), if `model_path` is not set, the default expected path is:

```text
<POLICY_ROOT>/<agent_run_name>/logs/<agent_run_name>-<policy>
```

## Notes

- This example is deployment-only (evaluation). No training happened in this example.
- `drl_step` controls when the controller is queried; between refreshes, the last action is held for the whole group.
- `u_tau` is used to normalize observations before calling the controller (the code comments note that solver-side normalization by `u_tau` should be kept consistent with how the policies were trained).
- This demo uses deterministic controller calls (`controller.predict(..., deterministic=True)`), and displays a reward summary to help compare controller configurations.

---

**Last Updated**: April 2026
**HydroGym Version**: 1.0+
**Maintainer**: HydroGym Team
