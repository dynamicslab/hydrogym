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
#     ./run_channel_docker.sh [mode] [num_steps] [dtype]
#
#     ./run_channel_docker.sh                              # no actuation, 5 steps, float32
#     ./run_channel_docker.sh drag_reduction               # small uniform suction
#     ./run_channel_docker.sh strong_actuation 20          # near-max jets, 20 steps
#     ./run_channel_docker.sh no_actuation 10 float64      # float64 precision
#
# ── Output ──────────────────────────────────────────────────────────────────
#
#   Per-step table: step | WSS | reward
#   WSS ≈ 0.0019 at Re_tau=180 in the uncontrolled case.
#   Drag reduction is achieved when WSS (and reward magnitude) decreases.
#

set -e

module purge
module load Python/3.12.3-GCCcore-13.3.0

# Activate Python environment
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-no_actuation}"
NUM_STEPS="${2:-5}"
DTYPE="${3:-float32}"

python "$SCRIPT_DIR/test_channel_env.py" "$MODE" --num-steps "$NUM_STEPS" --dtype "$DTYPE"
