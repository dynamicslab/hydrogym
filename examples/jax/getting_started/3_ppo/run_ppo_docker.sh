#!/usr/bin/env bash
#
# Run PPO training on a HydroGym JAX environment.
#
# All arguments are forwarded directly to run_ppo.py.
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
#     ./run_ppo_docker.sh                                  # Kolmogorov defaults
#     ./run_ppo_docker.sh --env channel                    # Channel flow
#     ./run_ppo_docker.sh --env kolmogorov --total-timesteps 20000
#     ./run_ppo_docker.sh --env channel --num-envs 2 --num-steps 5
#     ./run_ppo_docker.sh --help                           # Full argument list
#
# ── Output ────────────────────────────────────────────────────────────────────
#
#   trained_model_<env>.pkl   -- serialised Flax parameter tree
#   plot_reward_<env>.png     -- episode return vs. update steps
#   rewardovertime.npy        -- raw return array
#

set -e

module purge
module load Python/3.12.3-GCCcore-13.3.0

# Activate Python environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/run_ppo.py" "$@"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
