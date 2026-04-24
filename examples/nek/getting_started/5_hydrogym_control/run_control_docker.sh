#!/usr/bin/env bash
#
# Run NEK5000 integrate() control tests with MPMD coupling.
#
# Usage:
#     ./run_control_docker.sh                    # Test with default (OC) controller
#     ./run_control_docker.sh test OC            # Test with opposition control
#     ./run_control_docker.sh test BL            # Test with constant blowing
#     ./run_control_docker.sh test SIN           # Test with sinusoidal control
#     ./run_control_docker.sh test ZERO          # Test with zero control (baseline)
#     ./run_control_docker.sh train              # Train with RL then evaluate

set -e

# Load Nek5000 module
module purge
module load Nek5000/1.0-gompi-2024a-SystemCUDA-MiniChannel

# Activate Python environment
source ~/venvs/hydrogym_cpu/bin/activate

export OMP_NUM_THREADS=1

# Configuration
WORK_DIR="./train_run"
LOCAL_DIR="/workspace/hydrogym/packaged_envs"
ENV_NAME="TCFmini_3D_Re180"
NPROC_NEK=10
NUM_STEPS=100
EVAL_STEPS=200
TOTAL_TIMESTEPS=50000
MODE="${1:-test}"  # test or train
CONTROLLER="${2:-OC}"  # OC, BL, SIN, ZERO

echo "=== NEK5000 Control with integrate() ==="
echo "Mode: $MODE"
echo "Environment: $ENV_NAME"
echo "Nek5000 procs: $NPROC_NEK"
if [ "$MODE" != "train" ]; then
    echo "Controller: $CONTROLLER"
fi
echo ""

if [ "$MODE" == "train" ]; then
    echo "=== Training then Evaluating with integrate() ==="

    # Prepare workspace
    python ../prepare_workspace.py \
        --local-dir "$LOCAL_DIR" \
        --env "$ENV_NAME" \
        --work-dir "$WORK_DIR" \
        --cache-dir "$HOME/.cache/hydrogym"

    cd "$WORK_DIR" || exit 1

    mpirun --use-hwthread-cpus\
        -np 1 python ../train_sb3_with_integrate.py \
            --env "$ENV_NAME" \
            --local-dir "$LOCAL_DIR" \
            --nproc ${NPROC_NEK} \
            --total-timesteps ${TOTAL_TIMESTEPS} \
            --eval-steps ${EVAL_STEPS} \
            --algo PPO \
        : \
        -np ${NPROC_NEK} bash -c "nek5000 > /dev/null 2>&1"

else
    echo "=== Testing Controller: $CONTROLLER ==="
    echo "Using MAIA pattern with environment: $ENV_NAME"

    # Prepare workspace
    python ../prepare_workspace.py \
        --local-dir "$LOCAL_DIR" \
        --env "$ENV_NAME" \
        --work-dir "$WORK_DIR" \
        --cache-dir "$HOME/.cache/hydrogym"

    cd "$WORK_DIR" || exit 1

    mpirun --use-hwthread-cpus\
        -np 1 python ../test_nek_env_controller.py \
            --env "$ENV_NAME" \
            --nproc ${NPROC_NEK} \
            --local-dir "$LOCAL_DIR" \
            --steps ${NUM_STEPS} \
            --controller ${CONTROLLER} \
            --verbose \
        : \
        -np ${NPROC_NEK} bash -c "nek5000 > /dev/null 2>&1"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
