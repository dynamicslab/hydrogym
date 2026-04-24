#!/usr/bin/env bash
#
# Run Kolmogorov flow JAX environment for two control objectives:
#
#   Objective 1 -- Minimize TKE (suppress energy bursts)
#       reward = -(reward_alpha * TKE + action_penalty),  reward_alpha > 0
#       The agent is penalized for high turbulent kinetic energy and large
#       actions. Setting reward_alpha > 0 drives the flow toward a laminar,
#       low-energy state.
#
#   Objective 2 -- Maximize TKE (enhance turbulent mixing)
#       reward = -(reward_alpha * TKE + action_penalty),  reward_alpha < 0
#       A negative reward_alpha flips the sign of the TKE term, making the
#       reward proportional to TKE. The agent is rewarded for driving the
#       flow into a more turbulent regime while being penalized for large
#       actions.
#
#   In both cases the action penalty term (-sum(|a_i|)) discourages
#   unnecessarily large actuations and promotes efficient controllers.
#
# Usage:
#     ./run_kolmogorov_docker.sh [mode] [num_steps] [dtype]
#
#     ./run_kolmogorov_docker.sh                           # minimize TKE, 10 steps, float64
#     ./run_kolmogorov_docker.sh maximize_tke              # maximize TKE
#     ./run_kolmogorov_docker.sh no_actuation 500          # baseline, 500 steps
#     ./run_kolmogorov_docker.sh minimize_tke 1000 float32 # float32 (fast, may diverge)
#
# Actuation:
#     The control input is the amplitude of four sinusoidal body-force modes
#     added to the x-momentum equation:
#         c(y) = a1*sin(k1*y) + a2*sin(k2*y) + a3*sin(k3*y) + a4*sin(k4*y)
#     with wavenumbers k1,k2,k3,k4 = 4,5,6,7 (above the base forcing wavenumber).
#     Actions are clipped to [-0.5, 0.5].
#
# Precision:
#     float64 (default) -- required for JIT stability; matches non-JIT behavior
#     float32           -- faster but may produce NaNs due to solver instability under XLA
#

set -e

module purge
module load Python/3.12.3-GCCcore-13.3.0

# Activate Python environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-minimize_tke}"
NUM_STEPS="${2:-10}"
DTYPE="${3:-float64}"

python "$SCRIPT_DIR/test_kolmogorov_env.py" "$MODE" --num-steps "$NUM_STEPS" --dtype "$DTYPE"
