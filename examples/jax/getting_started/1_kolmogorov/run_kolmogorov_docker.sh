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
#     ./run_kolmogorov_docker.sh                      # Objective 1: minimize TKE
#     ./run_kolmogorov_docker.sh maximize_tke         # Objective 2: maximize TKE
#     ./run_kolmogorov_docker.sh no_actuation         # Baseline: zero action
#
# Actuation:
#     The control input is the amplitude of four sinusoidal body-force modes
#     added to the x-momentum equation:
#         c(y) = a1*sin(k1*y) + a2*sin(k2*y) + a3*sin(k3*y) + a4*sin(k4*y)
#     with wavenumbers k1,k2,k3,k4 = 4,5,6,7 (above the base forcing wavenumber).
#     Actions are clipped to [-0.5, 0.5].
#
# Output:
#     kolmogorov_<mode>.png  --  vorticity snapshots comparing baseline vs
#                                actuated trajectories
#

set -e

# Activate Python environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-minimize_tke}"
NUM_STEPS=10

echo "=== Kolmogorov Flow JAX Environment ==="
echo "Mode: $MODE"
echo "Steps per run: $NUM_STEPS"
echo ""

case "$MODE" in

  minimize_tke)
    echo "Objective: Minimize TKE (suppress energy bursts)"
    echo "  reward_alpha =  1.0  ->  reward = -(TKE + action_penalty)"
    echo "  Action: small forcing to damp energy transfer"
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})

# reward_alpha > 0: penalize TKE -> agent learns to suppress energy bursts
params = env.default_params.replace(reward_alpha=1.0)

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

action = jnp.array([-0.25, -0.03, 0.02, 0.01])

print(f"{'Step':>5}  {'mean_TKE':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    print(f"{i:>5}  {float(info['mean_tke']):>12.4f}  {float(reward):>12.4f}")
PYEOF
    ;;

  maximize_tke)
    echo "Objective: Maximize TKE (enhance turbulent mixing)"
    echo "  reward_alpha = -1.0  ->  reward = TKE - action_penalty"
    echo "  Action: forcing to drive the flow into a more turbulent regime"
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})

# reward_alpha < 0: reward proportional to TKE -> agent learns to increase mixing
params = env.default_params.replace(reward_alpha=-1.0)

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

action = jnp.array([0.25, 0.03, -0.02, -0.01])

print(f"{'Step':>5}  {'mean_TKE':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    print(f"{i:>5}  {float(info['mean_tke']):>12.4f}  {float(reward):>12.4f}")
PYEOF
    ;;

  no_actuation)
    echo "Baseline: zero actuation (free turbulence evolution)"
    echo "  reward_alpha = 1.0, action = [0, 0, 0, 0]"
    echo "  Shows natural energy bursts without control"
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

env = KolmogorovFlow(env_config={}, flow_config={})
params = env.default_params.replace(reward_alpha=1.0)

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

action = jnp.zeros((params.action_dim,))

print(f"{'Step':>5}  {'mean_TKE':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    print(f"{i:>5}  {float(info['mean_tke']):>12.4f}  {float(reward):>12.4f}")
PYEOF
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [minimize_tke|maximize_tke|no_actuation]"
    exit 1
    ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully."
else
    echo "Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
