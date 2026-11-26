# Zero-Shot Wing Deployment (Multi-Policy MARL)

Zero-shot deployment demo for the small NACA4412 wing case: multiple control policies are mapped to actuator subsets and executed together in one PettingZoo rollout.

__This is a deployment/evaluation demo only (no training). The template and controllers are intended for demonstration and should not be used to draw physical conclusions.__

> NOTE: The provided Nek5000 executable is pre-compiled for this chapter, so this demo focuses on the DRL-style rollout/deployment workflow.

> NOTE: Nek5000 reads the case name with its full path, which must be shorter than 132 characters. From a deeper
> working directory Nek5000 stops at start-up ("Cannot open .rea file!") and the Python rank then waits forever;
> run the demo from a short path (e.g. a copy of the examples in your home directory).

## What the script does

`zeroshot_demo_pettingzoo.py`:
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

- `zeroshot_demo_pettingzoo.py` - zero-shot multi-policy rollout demo (deployment only)
- `meta_policy_small_wing_template.py` - template defining `ENV_NAME`, `NPROC`, and `POLICY_SPECS`
- `run_pettingzoo_docker.sh` - launch script (image environment + workspace prep + `mpirun`)

## Usage

### Recommended: use the runner script
From `6_zeroshot_wing_demo/`:

```bash
./run_pettingzoo_docker.sh
./run_pettingzoo_docker.sh --policy-root /workspace/legacy_runs
```

### Direct: run the Python deployment script

Nek5000 reads the case from the current directory, so run the script from a prepared workspace (as the runner does).
Keep the workspace path short (see the note at the top).

Default template:
```bash
# Nek5000 executable of this case (in the HydroGym images; elsewhere your small_wing build)
export PATH=/opt/nek5000/cases/small_wing:$PATH
# Prepare the workspace once (copies the case files). Offline: add --local-dir DIR here and to the mpirun line
python ../prepare_workspace.py --env NACA4412_3D_Re75000_AOA5 --work-dir ./ws --profile NEK5000_v17
cd ws && rm -f *.sch
mpirun -np 1 python ../zeroshot_demo_pettingzoo.py --policy-template ../meta_policy_small_wing_template.py : -np 12 nek5000
```

Policy template + run root (from `ws/`):
```bash
mpirun -np 1 python ../zeroshot_demo_pettingzoo.py \
  --policy-template ../meta_policy_small_wing_template.py \
  --policy-root /path/to/legacy_runs \
  --steps 3000 \
  : -np 12 nek5000
```

Useful overrides:
- `--policy-template PATH` (defaults to `./meta_policy_small_wing_template.py` in the current directory)
- `--env ENV_NAME` (defaults from template `ENV_NAME`)
- `--nproc NPROC` (defaults from template `NPROC`)
- `--steps NUM_STEPS` (defaults from template `NUM_STEPS`)
- `--policy-root PATH` (where RL model run folders live)
- `--local-dir PATH` (optional fallback dir for packaged envs)
- `--log-every N` (reward table frequency)

## Policy Template (`meta_policy_small_wing_template.py`)

The template defines a multi-policy deployment configuration (one policy group per wall segment).

Required top-level variables:
- `ENV_NAME`
- `NPROC`
- `NUM_STEPS`
- `POLICY_ROOT` (default for `--policy-root`)
- `POLICY_SPECS` (list of policy group dicts)

Each `POLICY_SPECS` entry supports:
- `name`
- `x_range: [x_min, x_max]`
- `side: "SS"` (y>0) or `"PS"` (y<0)
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
- `u_tau` is an extra scaling of the observations before the controller is called (obs / u_tau); the Nek5000 side already scales observations and actions with its own u_tau, so keep `u_tau: 1.0` for policies trained with that scaling.
- This demo uses deterministic controller calls (`controller.predict(..., deterministic=True)`), and displays a reward summary to help compare controller configurations.
