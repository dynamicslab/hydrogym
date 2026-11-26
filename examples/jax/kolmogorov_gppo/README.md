# PPO and gradient-enhanced PPO (GPPO) on the Kolmogorov flow

Trainers, the released seed-30 policies, run commands for every seed and variant, an evaluation protocol, analysis
scripts, pinned versions and tests for PPO and GPPO on the Kolmogorov flow extreme-event preset, using HydroGym alone.

## Overview
This package trains both algorithms for any number of seeds (default: seeds 30-34), together with two control
variants (below), and evaluates the trained policies by deterministic roll-outs against baselines. The released
seed-30 PPO and GPPO policies are in `policies/kolmogorov_re40/{ppo,gppo}/seed30/` (`params.msgpack`, flax bytes).

## Environment
```python
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow.extreme_events_re40()  # preset KOLMOGOROV_EXTREME_EVENTS_RE40
```
- Re 40, forcing sin(4y) in x, [0, 2π]², 64 × 64 grid with the periodic end point (`legacy_grid_endpoint`), advection
  term not dealiased (`legacy_dealiasing`);
- 4 control modes a_i sin(k_i y) in x, k = 4, 5, 6, 7, actions in [−0.4, 0.15];
- dt 1e-3, 4 time units (4000 solver steps) per action, 4 saved snapshots per action, 100 actions per episode;
- observation: the window mean of the velocity magnitude on 8 × 8 points (64 values);
- reward −(5 · window-mean dissipation + 2 · Σ a_i²); the reset is deterministic (one unactuated window from a
  fixed initial condition);
- float32: `jax_enable_x64` stays off. Importing `hydrogym.jax.envs.channel` switches x64 on for the whole process,
  so do not import it in the same process.

**GPPO** adds β g·μ_θ(s) to the PPO loss, with β = 2. Here g is the unit gradient of the window-mean dissipation with
respect to the action, from `jax.jacfwd` through the solver (`env.rollout_objective`). It is computed for one
transition per update: env `(u // 10) % 4`, step `u % 10`.

## Variants
PPO and GPPO differ in the gradient term and in the advantage normalisation (PPO per minibatch, GPPO over the whole
rollout); two control variants separate the effects:

| id | command | normalisation | gradient term |
|---|---|---|---|
| `ppo_mb` | `python ppo.py --seed S --out-dir runs/ppo_mb/sS --resume` | minibatch | no (PPO) |
| `ppo_ro` | `python ppo.py --adv-norm rollout --seed S --out-dir runs/ppo_ro/sS --resume` | rollout | no |
| `gppo_b0` | `python gppo.py --beta 0 --seed S --out-dir runs/gppo_b0/sS --resume` | rollout | computed, weight 0 |
| `gppo_b2` | `python gppo.py --beta 2.0 --seed S --out-dir runs/gppo_b2/sS --resume` | rollout | yes (GPPO) |

Seeds S = 30 (the seed of the released policies), 31, 32, 33, 34; `--num-updates 100` is the default.
- Batches: 4 envs × 10 steps = 40 transitions per update; 100 updates = 4000 transitions = 40 episodes.
- `ppo_ro` and `gppo_b0` are the same algorithm in exact arithmetic. Their logs agree to float round-off for the
  first updates, then drift apart (chaotic flow).
- With the same seed, all four variants start from the same initial network and the same first batch.
- Options: `--resume` continues bit for bit from `<out-dir>/<algo>_ckpt.pkl`; `--max-minutes M` stops cleanly with
  exit code 3 for time-limited jobs; `<out-dir>/<algo>_DONE` marks a finished run.
- Outputs: `<algo>_metrics.npz` (per update: reward, dissipation, control cost, loss, `wall_s`; GPPO also
  `gradient_norm` and `grad_wall_s`) and `<algo>_params.pkl` (flax bytes).

## How to run
- **Local** (one GPU, or a CPU at ~2–3 min per update), from this directory:
  `bash run_all.sh /path/to/run_dir`. It trains all 20 runs (seeds and variants can be set through `SEEDS=...`
  and `VARIANTS=...`), then runs `evaluate.py` and `analyse.py`. Re-running it resumes.
- **Slurm** (≤ 1 h jobs, self-resubmitting, one run per GPU):
  1. Add the `#SBATCH` partition / account lines your cluster requires to `slurm/*.sbatch`.
  2. Run `KGP_RUN_DIR=/path/to/run_dir KGP_PYTHON=/path/venv/bin/python bash slurm/submit_all.sh`. This submits one
     chain per seed (4 variants per 4-GPU node) and the main evaluation.
  3. After training:
     `KGP_RUN_DIR=... KGP_EXAMPLE_DIR=$PWD sbatch --export=ALL --job-name=kgp_ev_tr slurm/eval_chain.sbatch 1 trained`.
  4. Then run `python analyse.py $KGP_RUN_DIR $KGP_RUN_DIR/analysis`.
- **Runtime** (one GPU: about 40 s per update, 1 run per GPU; several runs on one GPU give no throughput gain):
  - about 70 min per 100-update run;
  - main evaluation about 25 min on 4 GPUs.
- **Versions:** `requirements.txt` has the key pins; `requirements-lock.txt` is the pinned runtime environment
  (CPython 3.12, jax 0.11.2 with the CUDA 13 plugin). HydroGym is on `PYTHONPATH`.
- **Repeatability:** the flow is chaotic. On a GPU, XLA may choose other kernels in every new process, so two runs
  of the same command can differ; `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes GPU runs repeatable, and
  `JAX_PLATFORMS=cpu` runs on the CPU, where repeated runs are bit-identical. GPU and CPU runs give different
  numbers.

## Evaluation protocol (`evaluate.py`) and analysis (`analyse.py`)
- **Uncontrolled reference:** 8 trajectories of 500 windows (2000 time units). Trajectory j starts from the reset
  with a constant action of amplitude 0.02 along a fixed random direction for the first window; j = 0 is
  unperturbed. Statistics are taken from t ≥ 500, per snapshot (one per time unit). The extreme-event threshold is
  mean + 3 sd of the snapshot dissipation.
- **Start states:** a bank of 16 states, windows 150 and 250 of each trajectory, plus the training reset state.
- **Policies:** the released PPO / GPPO seed-30 policies, deterministic. GPPO and PPO are also run stochastically
  from the reset with 16 noise seeds.
- **Baselines:**
  - uncontrolled;
  - each policy's mean action applied as a constant;
  - the best constant on a 5⁴ grid in [−0.4, 0.15]⁴: screened on 4 bank states over 25 actions, then the best
    8 by reward and the best 4 by dissipation on the whole bank over 100 actions.
- **Trained policies:** every run in `runs/`, deterministic, from the bank and from the reset.
- **Metrics per episode:** mean dissipation, reward, control cost, TKE, actuation power (signed and absolute),
  forcing power, extreme-event rate and mean excess.

**Measures** (`analyse.py`):
- **Curves and final performance:** the batch reward, dissipation and control cost per update, smoothed by a
  trailing 10-update mean (MA10). The final performance is the mean over updates 91–100.
- **S1, the reward sample-efficiency measure (primary):**
  - R* = the mean final reward of `ppo_mb`;
  - U = the first update at which the seed-mean MA10 reward reaches R*;
  - reduction = 1 − U_X / U_ppo_mb, with a bootstrap over seeds.
- **S2, the dissipation measure:** D* = 0.9 × the mean first-update dissipation, and the first update at which
  MA10 ≤ D*.
- **Decomposition:**
  - total effect = `gppo_b2` − `ppo_mb`;
  - normalisation effect = N − `ppo_mb`;
  - gradient-term effect = `gppo_b2` − N.

  N is `ppo_ro` ∪ `gppo_b0` if their final rewards do not differ (Welch p > 0.05), otherwise `gppo_b0` alone. All
  effects come with 95 % bootstrap CIs and Welch tests.

## Tests
- `test/test_kolmogorov_gppo_package.py` runs on a CPU with tiny settings. It covers:
  - the four variants training end to end;
  - `evaluate.py --quick` with every task, including the trained runs;
  - `analyse.py` on synthetic logs.
- Related tests:
  - `test/test_kolmogorov_gppo_resume.py`: resume is exact, and the controls;
  - `test/test_jax_kolmogorov_preset.py`: the preset;
  - `test/test_jax_solver_step_count.py`: the step count.
