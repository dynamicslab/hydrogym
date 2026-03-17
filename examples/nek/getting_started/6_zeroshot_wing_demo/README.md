# Zero-Shot Wing Deployment (Multi-Policy MARL)

Demonstration of zero-shot deployment where multiple control policies are mapped
to different wing regions and executed together in one PettingZoo rollout.

## Interface

```python
from hydrogym.nek import NekEnv, make_pettingzoo_env

base_env = NekEnv.from_hf("NACA4412_3D_Re75000_AOA5", nproc=12)
env = make_pettingzoo_env(base_env)

obs, info = env.reset()
actions = {agent: policy(obs[agent]) for agent in env.agents}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

## Files

- `test_nek_pettingzoo.py` - zero-shot multi-policy rollout demo (deployment only)
- `meta_policy_small_wing_template.py` - template matching old MetaPolicy concepts
- `run_pettingzoo_docker.sh` - Docker/MPI runner for this chapter

## Usage

### Run Zero-Shot Demo (default template)
```bash
mpirun -np 1 python test_nek_pettingzoo.py : -np 12 nek5000
```

### Run with Legacy RL Models
```bash
mpirun -np 1 python test_nek_pettingzoo.py \
    --policy-template ./meta_policy_small_wing_template.py \
    --policy-root /path/to/legacy_runs \
    --steps 3000 \
    : -np 12 nek5000
```

## Mapping from Old Framework

This chapter keeps the same core ideas as your old `eval_meta.py` + `MetaPolicy.py`:

- control-region assignment by `x_range` and `side` (`SS` / `PS`)
- mixed policy types (`PPO`/`TD3`/`DDPG`/`BL`/`OC`/`ZERO`)
- per-group action update interval (`drl_step`)
- local scaling entries (`u_tau`, `baseline_dudy`)

## Legacy Model Path Convention

For RL policies (`PPO`, `TD3`, `DDPG`), the default expected model path is:

```text
<POLICY_ROOT>/<agent_run_name>/logs/<agent_run_name>-<policy>
```

You can also set `model_path` per policy entry in the template.

## Notes

- This is a deployment/evaluation demo, not a training script.
- All changes are isolated to this chapter directory.
- `NACA4412_3D_Re75000_AOA5` is the default environment for this demo.
