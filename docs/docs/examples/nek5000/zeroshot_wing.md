---
sidebar_position: 7
---

# Zero-Shot Wing Deployment (Multi-Policy MARL)

Zero-shot deployment demo for the small NACA4412 wing case: multiple control policies are mapped to actuator subsets and executed together in one PettingZoo rollout.

**This is a deployment/evaluation demo only (no training). The template and controllers are intended for demonstration and should not be used to draw physical conclusions.**

> NOTE: The provided Nek5000 executable is pre-compiled for this chapter, so this demo focuses on the DRL-style rollout/deployment workflow.

## What the script does

`test_nek_pettingzoo.py`:
- loads a base `NekEnv` via `NekEnv.from_hf(...)` and wraps it with `make_pettingzoo_env(...)`
- builds one controller per entry in `POLICY_SPECS` (from `meta_policy_small_wing_template.py`)
- assigns each controller to actuator agents by `x_range` and `side` (`SS` means `y > 0`, `PS` means `y < 0`)
- refreshes each group's actions every `drl_step` steps (refresh at step `0`; otherwise actions are held)
- clips actions to `action_bounds`
- computes an “inverted” + scaled reward summary for display (deployment-only)

Unassigned actuator agents receive zero action.

## Interface (PettingZoo rollout)

```python
from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)

obs_dict, info = env.reset()
actions = {agent: controller(obs_dict[agent]) for agent in env.agents}
obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
```

## Files

- `test_nek_pettingzoo.py` - zero-shot multi-policy rollout demo (deployment only)
- `meta_policy_small_wing_template.py` - template defining `ENV_NAME`, `NPROC`, and `POLICY_SPECS`
- `run_pettingzoo_docker.sh` - runner script (module load + workspace prep + `mpirun`)

## Usage

### Recommended: use the runner script
From `6_zeroshot_wing_demo/`:

```bash
./run_pettingzoo_docker.sh
./run_pettingzoo_docker.sh --policy-root /workspace/legacy_runs
```

### Direct: run the Python deployment script

Default template:
```bash
mpirun -np 1 python test_nek_pettingzoo.py : -np 12 nek5000
```

Legacy policy template + run root:
```bash
mpirun -np 1 python test_nek_pettingzoo.py \
  --policy-template ./meta_policy_small_wing_template.py \
  --policy-root /path/to/legacy_runs \
  --steps 3000 \
  : -np 12 nek5000
```

Useful overrides:
- `--policy-template PATH` (defaults to `./meta_policy_small_wing_template.py`)
- `--env ENV_NAME` (defaults from template `ENV_NAME`)
- `--nproc NPROC` (defaults from template `NPROC`)
- `--steps NUM_STEPS` (defaults from template `NUM_STEPS`)
- `--policy-root PATH` (where RL model run folders live)
- `--local-dir PATH` (optional fallback dir for packaged envs)
- `--log-every N` (reward table frequency)

## Policy Template (`meta_policy_small_wing_template.py`)

The template defines a lightweight legacy-`MetaPolicy.py`-style configuration.

Required top-level variables:
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

- Deployment-only (evaluation). No training happens in this chapter.
- `drl_step` controls when the controller is queried; between refreshes, the last action is held for the whole group.
- `u_tau` is used to normalize observations before calling the controller (the code comments note that solver-side normalization by `u_tau` should be kept consistent with how the policies were trained).
- This demo uses deterministic controller calls (`controller.predict(..., deterministic=True)`), and displays a reward summary to help compare controller configurations.

