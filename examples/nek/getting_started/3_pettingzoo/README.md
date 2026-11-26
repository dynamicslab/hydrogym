# NEK5000 PettingZoo Interface

PettingZoo-compatible wrapper for ecosystem integration and production-ready SB3 training via SuperSuit.

## Interface

```python
from hydrogym.nek import NekEnv, make_pettingzoo_env

# Create base environment
env_config = {"environment_name": "TCFmini_3D_Re180", "nproc": 10}
base_env = NekEnv(env_config=env_config)

# Wrap with PettingZoo interface
env = make_pettingzoo_env(base_env)

# PettingZoo parallel API
observations, infos = env.reset()
actions = {agent: policy(obs) for agent, obs in observations.items()}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

## Files

- `test_nek_pettingzoo.py` - Test PettingZoo interface
- `train_sb3_pettingzoo.py` - **SB3 training with SuperSuit** (production approach)
- `run_pettingzoo_docker.sh` - MPMD launch script (runs as is in the HydroGym images)

## Usage

### Test Environment
```bash
# from a prepared workspace: the Nek5000 case files must be in the working directory
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ws && cd ws
mpirun -np 1 python ../test_nek_pettingzoo.py --steps 100 : -np 10 nek5000
```

### Train RL Agent (SuperSuit Production Approach)
```bash
# in the workspace ws/ of the block above
mpirun -np 1 python ../train_sb3_pettingzoo.py --env TCFmini_3D_Re180 --algo PPO --total-timesteps 100000 : -np 10 nek5000
```

## Configuration-Driven Tutorial (Recommended for Reproducibility)

Use a fixed YAML config to lock simulation + runner settings across runs.

### 1) Prepare a workspace
```bash
python ../prepare_workspace.py \
  --env TCFmini_3D_Re180 \
  --work-dir ./train_run
```

### 2) Train with a config file
```bash
cd train_run
mpirun -np 1 python ../train_sb3_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../../../configs/pettingzoo_tcfmini_re180.yml \
  --algo TD3 \
  --total-timesteps 5000000 \
  : -np 10 nek5000
```

### 3) Roll out (PettingZoo API, zero control)
```bash
cd train_run
mpirun -np 1 python ../test_nek_pettingzoo.py \
  --env TCFmini_3D_Re180 \
  --nproc 10 \
  --config-file ../../../configs/pettingzoo_tcfmini_re180.yml \
  --steps 2500 \
  : -np 10 nek5000
```

Notes:
- The config lives in `examples/nek/configs/pettingzoo_tcfmini_re180.yml`.
- Run from the workspace (`train_run`) so `compile_path: "."` resolves to case files.
- Ensure `--nproc` matches `simulation.nproc` in the config.

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

### Comparison with Chapter 2

| Aspect | Chapter 2 (DIY) | Chapter 3 (SuperSuit) |
|--------|-----------------|----------------------|
| Purpose | Educational | Production |
| Dependencies | None extra | pettingzoo, supersuit |
| Code complexity | ~50 lines wrapper | ~5 lines SuperSuit |
| Maintenance | DIY | Ecosystem-maintained |
| Use when | Learning | Deploying |

## Notes

- SuperSuit is the **recommended** approach for production
- Chapter 2 (DIY wrapper) is for understanding how it works
- PettingZoo API ensures compatibility with many MARL tools

## Training settings and the shared-policy wrapper

`train_sb3_pettingzoo.py` reads every algorithm setting from the `runner` section of the environment configuration
(`batch_size`, `buffer_size`, `seed`, `learning_rate`, `learning_starts`, `train_steps`, `gradient_steps`, `tau`,
`policy_delay`, `target_policy_noise`, `target_noise_clip`, `action_noise`, `nb_interactions`, `nb_episodes`,
`ckpt_int`). Command-line options override a value only when given. Network defaults: actor [8], critic [16, 16, 64]
(`--actor-arch`, `--critic-arch`). The resolved values and where each came from are
written to `<log_dir>/resolved_settings.json`. The TD3 policy of the HydroGym paper
(`policies/naca0012_wing_re2e5/td3/seed1998`) has actor [8] and critic [16, 64, 64] (`--critic-arch 16 64 64`); the
`data` file of its zip records 4410 agents, gamma 0.99, batch size 2048, a replay buffer of 2·10⁷ and learning rate
10⁻³. It was trained with Stable-Baselines3 1.7 and Gym 0.24; this script uses current Stable-Baselines3 and
Gymnasium.

By default (`--wrapper shared`) the script uses `hydrogym.nek.shared_policy.make_shared_policy_vec_env`: one agent per
physically distinct wall point, array-based and linear in the number of agents. GLL nodes on element edges belong to
two (at corners four) elements, and on a periodic wall the nodes at both ends coincide; the shared wrapper trains on
each physical point once (`TCFlarge_3D_Re207`: 4410 agents from 5760 wall nodes). `--keep-duplicates` restores one
agent per node, and `--wrapper supersuit` selects the PettingZoo + SuperSuit route (one agent per GLL node; about 50
times slower for the channel).

Note on the action interval of `TCFlarge_3D_Re207`: `ndrl = 13` solver steps of `dt = 4e-3` at `Re_b = 3300` and
`u_tau = 0.063` give `dt+ = 13 * 4e-3 * 0.063**2 * 3300 = 0.68`.
