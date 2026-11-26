#!/usr/bin/env bash
#
# Run PPO training on a HydroGym JAX environment.
#
# All arguments are forwarded directly to run_ppo.py.
#
# Runs as is in the HydroGym container images (everything on PATH);
# elsewhere, activate a Python environment with HydroGym first.
#
# ── Supported environments ────────────────────────────────────────────────────
#
#   kolmogorov  -- 2D Kolmogorov flow (Re=200, 64×64 pseudo-spectral)
#                  Action: 4 sinusoidal body-force amplitudes (wavenumbers 4-7)
#                  Observation: 8×8 = 64 velocity probe values
#                  Reward: -(reward_alpha * TKE + action_penalty)
#
#   channel     -- 3D turbulent channel flow (Re_tau=180, 72×72×72 DNS)
#                  Action: 24 wall-normal jet amplitudes (6×4 Gaussian patches)
#                  Observation: 8×8×2 = 128 near-wall velocity samples
#                  Reward: -WSS  (minimize wall shear stress / skin-friction drag)
#                  Note: significantly more expensive per step than Kolmogorov.
#                  The initial turbulent field is downloaded from HuggingFace on
#                  the first run and cached at ~/.cache/hydrogym/.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#
# The defaults are getting-started settings that run on a laptop CPU: short
# episodes (Kolmogorov: 20 actions of 1 time unit at dt 5e-3; channel: 10
# actions of 50 DNS steps) and few timesteps (Kolmogorov 4000, channel 120).
#
#     ./run_ppo_docker.sh                                  # Kolmogorov, ~10-15 min on a CPU
#     ./run_ppo_docker.sh --env channel                    # Channel flow, ~15-20 min on a CPU
#     ./run_ppo_docker.sh --help                           # Full argument list
#
# Full-size runs (GPU recommended) use the environments' own episodes
# (Kolmogorov: 1000 actions of 10 time units at dt 1e-3; channel: 100 actions
# of 50 DNS steps) via --env-defaults, with more timesteps, e.g. (about a day
# and half a day on a CPU, respectively)
#
#     ./run_ppo_docker.sh --env kolmogorov --env-defaults --total-timesteps 20000 --num-steps 100
#     ./run_ppo_docker.sh --env channel --env-defaults --total-timesteps 5000 --num-steps 50
#
# ── Output ────────────────────────────────────────────────────────────────────
#
#   trained_model_<env>.pkl   -- serialised Flax parameter tree
#   plot_reward_<env>.png     -- mean reward per step for each PPO update with
#                                its running mean (top) and the returns of the
#                                episodes that ended in each update (bottom)
#   rewardovertime.npy        -- mean reward per step for each PPO update
#

set -e

# Python environment: the HydroGym container images have it in /opt/venvs;
# elsewhere, activate your own environment before running this script.
if [ -f /opt/venvs/hydrogym/bin/activate ]; then source /opt/venvs/hydrogym/bin/activate; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# With only 2-3 CPUs (e.g. a container started with --cpus/--cpuset-cpus 2-3, or srun -c 2), the XLA CPU thread pool
# can deadlock in the channel's first compiled step. Its single-threaded Eigen path avoids that. (The images set
# OMP_NUM_THREADS=1, which plain nproc would report.)
if [ "$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)" -lt 4 ] && [[ "${XLA_FLAGS:-}" != *xla_cpu_multi_thread_eigen* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false"
fi


# Run training. `|| EXIT_CODE=$?` captures a nonzero status without tripping
# `set -e`, so the reporting block below always executes.
EXIT_CODE=0
python "$SCRIPT_DIR/run_ppo.py" "$@" || EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
