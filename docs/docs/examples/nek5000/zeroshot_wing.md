---
sidebar_position: 7
---

# Zero-Shot Wing Deployment

This demo shows how to deploy multiple pre-trained control policies simultaneously on the NACA4412 small-wing NEK5000 case using a PettingZoo rollout. Different policies are mapped to spatially distinct subsets of the wing actuators by chordwise position and pressure/suction side, and executed together in a single simulation without any additional training.

:::warning[Deployment only]
This is an evaluation demo. No training is performed. The controllers and template values are illustrative and should not be used to draw physical conclusions about the wing case.
:::

All scripts live in [`examples/nek/getting_started/6_zeroshot_wing_demo/`](https://github.com/dynamicslab/hydrogym/tree/main/examples/nek/getting_started/6_zeroshot_wing_demo).

## What the demo does

[`zeroshot_demo_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/zeroshot_demo_pettingzoo.py):

1. Loads a base [`NekEnv`](../../api/nek/env) via `NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)` and wraps it with [`make_pettingzoo_env`](../../api/nek/pettingzoo_env).
2. Reads the `POLICY_SPECS` list from the policy template (default: `meta_policy_small_wing_template.py`).
3. Assigns each policy entry to the actuator agents whose chordwise coordinate falls within `x_range` and whose side matches `"SS"` (suction side, y > 0) or `"PS"` (pressure side, y < 0).
4. Queries each controller every `drl_step` environment steps; between refreshes the last action is held constant for the assigned actuator group.
5. Clips all actions to `action_bounds` before passing them to the environment.
6. Prints a reward summary table at the end to help compare controller configurations.

Actuator agents not assigned to any policy entry receive a zero action throughout the rollout.

## PettingZoo rollout interface

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)

obs_dict, info = env.reset()
actions = {agent: controller(obs_dict[agent]) for agent in env.agents}
obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
```

## Example scripts

| File | Purpose |
|------|---------|
| [`zeroshot_demo_pettingzoo.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/zeroshot_demo_pettingzoo.py) | Multi-policy rollout demo |
| [`meta_policy_small_wing_template.py`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/meta_policy_small_wing_template.py) | Template defining `ENV_NAME`, `NPROC`, and `POLICY_SPECS` |
| [`run_pettingzoo_docker.sh`](https://github.com/dynamicslab/hydrogym/blob/main/examples/nek/getting_started/6_zeroshot_wing_demo/run_pettingzoo_docker.sh) | Docker/MPI launcher |

## Running the demo

The runner script is the recommended entry point — it handles module loading and workspace preparation automatically:

```bash
# Default template
./run_pettingzoo_docker.sh

# Specify a root directory containing pre-trained model runs
./run_pettingzoo_docker.sh --policy-root /workspace/legacy_runs
```

To invoke the Python script directly:

```bash
# Default template
mpirun -np 1 python zeroshot_demo_pettingzoo.py : -np 12 nek5000

# Custom template and policy root
mpirun -np 1 python zeroshot_demo_pettingzoo.py \
  --policy-template ./meta_policy_small_wing_template.py \
  --policy-root /path/to/legacy_runs \
  --steps 3000 \
  : -np 12 nek5000
```

Available command-line overrides:

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-template PATH` | `./meta_policy_small_wing_template.py` | Path to the policy template |
| `--env ENV_NAME` | From template `ENV_NAME` | Override the environment name |
| `--nproc NPROC` | From template `NPROC` | Number of solver processes |
| `--steps NUM_STEPS` | From template `NUM_STEPS` | Total rollout steps |
| `--policy-root PATH` | From template `POLICY_ROOT` | Root directory for RL model run folders |
| `--local-dir PATH` | — | Optional local fallback for packaged environments |
| `--log-every N` | — | Reward table print frequency |

## Policy template format

`meta_policy_small_wing_template.py` defines the following top-level variables:

| Variable | Type | Description |
|----------|------|-------------|
| `ENV_NAME` | `str` | Environment name for `NekEnv.from_hf()` |
| `NPROC` | `int` | Number of NEK5000 solver processes |
| `NUM_STEPS` | `int` | Total rollout length |
| `POLICY_ROOT` | `str` | Default root directory for RL model checkpoints |
| `POLICY_SPECS` | `list[dict]` | List of policy group definitions |

Each entry in `POLICY_SPECS` supports:

| Key | Description |
|-----|-------------|
| `name` | Human-readable label for the policy group |
| `x_range` | `[x_min, x_max]` — chordwise range of assigned actuators |
| `side` | `"SS"` (suction side, y > 0) or `"PS"` (pressure side, y < 0) |
| `algorithm` | `"PPO"`, `"TD3"`, `"DDPG"`, `"BL"` (constant max action), or `"ZERO"` |
| `drl_step` | Action refresh interval; actions are held between refreshes |
| `action_bounds` | `[min, max]` — clipping applied to controller output |
| `u_tau` | Optional: friction velocity used to normalise observations before calling the controller |
| `baseline_dudy` | Optional: baseline velocity gradient for normalisation |
| `agent_run_name` | RL only: identifier for the training run folder |
| `policy` | RL only: checkpoint identifier within the run folder |
| `model_path` | RL only: explicit path to the model file (overrides the default convention) |

### Default RL model path convention

When `model_path` is not set, the model is resolved from:

```
<POLICY_ROOT>/<agent_run_name>/logs/<agent_run_name>-<policy>
```

For overlapping `x_range` regions, the last-listed policy in `POLICY_SPECS` takes precedence.
