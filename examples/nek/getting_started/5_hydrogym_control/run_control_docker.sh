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

# Python environment: the HydroGym container images have it in /opt/venvs;
# elsewhere, activate your own environment before running this script.
if [ -f /opt/venvs/hydrogym/bin/activate ]; then source /opt/venvs/hydrogym/bin/activate; fi
# Nek5000 case executable: in the images /opt/nek5000/cases/mini_channel/nek5000;
# elsewhere, put the directory of your mini_channel build on PATH.
if [ -d /opt/nek5000/cases/mini_channel ]; then export PATH=/opt/nek5000/cases/mini_channel:$PATH; fi

export OMP_NUM_THREADS=1

# Configuration
WORK_DIR="./train_run"
# Local copy of the environment files, used when the Hugging Face Hub is
# unreachable (compute nodes without internet; see prepare_workspace.py).
LOCAL_DIR="${LOCAL_DIR:-$PWD/packaged_envs}"
# Extra mpirun options, e.g. MPIRUN_ARGS=--oversubscribe on machines with
# fewer cores than MPI ranks (Open MPI 5 counts slots by physical core).
MPIRUN_ARGS="${MPIRUN_ARGS:-}"
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

    mpirun --use-hwthread-cpus ${MPIRUN_ARGS} \
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
    echo "Using NekEnv.from_hf with environment: $ENV_NAME"

    # Prepare workspace
    python ../prepare_workspace.py \
        --local-dir "$LOCAL_DIR" \
        --env "$ENV_NAME" \
        --work-dir "$WORK_DIR" \
        --cache-dir "$HOME/.cache/hydrogym"

    cd "$WORK_DIR" || exit 1

    mpirun --use-hwthread-cpus ${MPIRUN_ARGS} \
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
