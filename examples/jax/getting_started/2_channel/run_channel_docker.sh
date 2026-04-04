#!/usr/bin/env bash
#
# Run 3D turbulent channel flow JAX environment (Re_tau = 180).
#
# ── Flow physics ────────────────────────────────────────────────────────────
#
#   This environment simulates incompressible turbulent channel flow between
#   two parallel walls separated by a distance Lz = 2 (wall-units: z ∈ [-1,1]).
#   The friction Reynolds number Re_tau = u_tau * delta / nu = 180, where:
#       u_tau  = friction velocity (set by the mean pressure gradient)
#       delta  = channel half-height = 1
#       nu     = kinematic viscosity = 1.9e-3
#
#   A constant streamwise body force (fx = 2.0) drives the flow against
#   viscous drag, maintaining the target bulk velocity u_bulk ≈ 8.
#
# ── DNS solver ──────────────────────────────────────────────────────────────
#
#   The flow is resolved with a pseudo-spectral Navier-Stokes solver:
#       x, y (streamwise, spanwise):  Fourier basis  (Nx=72, Ny=72 modes)
#       z    (wall-normal):           Chebyshev basis (Nz=72 points)
#
#   Each environment step advances the flow by nsteps=50 RK4 sub-steps at
#   dt=2e-4, i.e. Δt_RL = 0.01 convective time units per RL step.
#
#   Incompressibility is enforced after each RK4 step via a pressure-Poisson
#   projection on the divergence field.
#
# ── Actuation ───────────────────────────────────────────────────────────────
#
#   Wall-normal blowing/suction jets are applied at both walls through a
#   6 × 4 grid of Gaussian patches (nx_jets=6, ny_jets=4 → action_dim=24).
#   Each jet applies a smooth wall-normal velocity profile (w) to a thin
#   layer near the wall (z0=1, thickness=5 grid points).
#   Actions are clipped to [-1, 1].
#
#   The jets are mass-flux neutral: the mean is subtracted before application
#   (jnp.mean(mask) is removed), so no net mass is added to the domain.
#
#   A gain ramp (gain ≈ 0.3) smoothly switches actuation on/off at the start
#   and end of each episode to avoid impulsive forcing.
#
# ── Observation ─────────────────────────────────────────────────────────────
#
#   The agent observes a subsampled (8×8) grid of streamwise (U) and wall-
#   normal (W) velocity at a fixed wall-normal plane z = z[k_det=9], i.e.
#   near-wall but not at the wall itself.
#   Observation dimension: 8 × 8 × 2 = 128.
#
# ── Reward ──────────────────────────────────────────────────────────────────
#
#   reward = -WSS
#
#   where WSS (wall shear stress) is the viscous stress at the lower wall:
#       WSS = nu * (∂U/∂z)|_{z=wall}  averaged over x-y
#
#   Minimizing WSS ↔ reducing skin-friction drag. This is the primary
#   control objective for turbulent drag reduction.
#
# ── Initial condition ───────────────────────────────────────────────────────
#
#   The environment downloads a fully turbulent initial field (U.npy, V.npy,
#   W.npy) from HuggingFace (dynamicslab/HydroGym-environments,
#   Channel_3D_Retau180/initial_field/) on the first run. Subsequent runs
#   use the local cache at ~/.cache/hydrogym/.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#     ./run_channel_docker.sh                    # Baseline: no actuation
#     ./run_channel_docker.sh drag_reduction     # Small uniform suction
#     ./run_channel_docker.sh strong_actuation   # Near-max amplitude jets
#
# ── Output ──────────────────────────────────────────────────────────────────
#
#   Per-step table: step | WSS | reward
#   WSS ≈ 0.0019 at Re_tau=180 in the uncontrolled case.
#   Drag reduction is achieved when WSS (and reward magnitude) decreases.
#

set -e

# Activate Python environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-no_actuation}"
NUM_STEPS=5      # RL steps (each = 50 DNS sub-steps at dt=2e-4)

echo "=== 3D Turbulent Channel Flow JAX Environment (Re_tau=180) ==="
echo "Mode:      $MODE"
echo "RL steps:  $NUM_STEPS  (= $((NUM_STEPS * 50)) DNS sub-steps, Δt_DNS=2e-4)"
echo ""

case "$MODE" in

  no_actuation)
    echo "Baseline: zero actuation — free turbulent evolution"
    echo "  action = 0 (all 24 jet amplitudes set to zero)"
    echo "  Shows natural WSS fluctuations without control."
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

# HF download happens here on first run (cached afterwards)
env = ChannelFlowSpectralEnv(env_config={})
params = env.default_params

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

# Zero action: no wall jets, passive turbulence evolution
action = jnp.zeros((params.action_dim,))

print(f"{'Step':>5}  {'WSS':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    # reward = -WSS, so WSS = -reward
    print(f"{i:>5}  {float(-reward):>12.6f}  {float(reward):>12.6f}")
PYEOF
    ;;

  drag_reduction)
    echo "Drag reduction: small uniform suction at the walls"
    echo "  action = 0.01 (uniform positive amplitude across all 24 jets)"
    echo "  Gentle blowing/suction perturbs near-wall streaks to reduce drag."
    echo "  Expected: WSS decreases relative to the no-actuation baseline."
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

env = ChannelFlowSpectralEnv(env_config={})
params = env.default_params

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

# Uniform low-amplitude suction across all jet locations.
# The mean is subtracted internally (mass-flux neutral), so this
# effectively creates a spatially uniform suction pattern.
action = 0.01 * jnp.ones((params.action_dim,))

print(f"{'Step':>5}  {'WSS':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    print(f"{i:>5}  {float(-reward):>12.6f}  {float(reward):>12.6f}")
PYEOF
    ;;

  strong_actuation)
    echo "Strong actuation: alternating high-amplitude jets"
    echo "  action alternates ±0.5 across the 6×4 jet grid"
    echo "  Stress-tests the actuator: large perturbations to near-wall flow."
    echo "  NOTE: very large actions may destabilize the simulation."
    echo ""
    python - <<PYEOF
import jax
import jax.numpy as jnp
from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

env = ChannelFlowSpectralEnv(env_config={})
params = env.default_params

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

# Checkerboard pattern: alternating +0.5 / -0.5 across the 6×4 jet grid.
# After mean subtraction inside apply_jets_v this creates a spanwise-varying
# forcing that targets the large-scale near-wall streaks.
nx_jets, ny_jets = 6, 4
i_idx = jnp.arange(nx_jets)[:, None]
j_idx = jnp.arange(ny_jets)[None, :]
pattern = jnp.where((i_idx + j_idx) % 2 == 0, 0.5, -0.5)
action = pattern.reshape(-1)   # shape (24,)

print(f"{'Step':>5}  {'WSS':>12}  {'reward':>12}")
print("-" * 35)
for i in range($NUM_STEPS):
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(subkey, state, action, params)
    print(f"{i:>5}  {float(-reward):>12.6f}  {float(reward):>12.6f}")
PYEOF
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [no_actuation|drag_reduction|strong_actuation]"
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
