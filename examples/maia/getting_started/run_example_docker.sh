#!/usr/bin/env bash
#
# Run MAIA single-agent tests and training with MPMD coupling.
#
# Usage:
#     ./run_example_docker.sh                    # Test only
#     ./run_example_docker.sh train              # Train SB3 agent

set -e

# Load MAIA solver module
# GPU version: MAIA/1.0-NVHPC-26.1
# CPU version: MAIA/1.0-gompi-2024a-SystemCUDA-CPU
module purge
module load MAIA/1.0-NVHPC-26.1

# Activate Python environment
# GPU version: source /home/easybuild/venvs/hydrogym_gpu/bin/activate
# CPU version: source /home/easybuild/venvs/hydrogym_cpu/bin/activate
source /home/easybuild/venvs/hydrogym_gpu/bin/activate

export OMP_NUM_THREADS=1

# Configuration
WORK_DIR="./train_run"
ENVIRONMENT="Cylinder_2D_Re200"
PROPERTIES_FILE="properties_run.toml"
NPROC_MAIA=1   # num GPUs or cpu cores
NUM_STEPS=100
TOTAL_TIMESTEPS=50000
MODE="${1:-test}"  # test or train

echo "=== MAIA Single Agent ==="
echo "Mode: $MODE"
echo "Environment: $ENVIRONMENT"
echo "MAIA procs: $NPROC_MAIA"
echo ""

if [ "$MODE" == "train" ]; then
    echo "=== Training SB3 Agent ==="

    # Prepare workspace
    python prepare_workspace.py \
        --env "$ENVIRONMENT" \
        --work-dir "$WORK_DIR"

    cd "$WORK_DIR" || exit 1

    echo "Work directory: $(pwd)"
    echo "Properties file: $PROPERTIES_FILE"
    echo ""

    mpirun \
        -np 1 python ../train_sb3_maia.py \
            --env "$ENVIRONMENT" \
            --total-timesteps ${TOTAL_TIMESTEPS} \
            --algo PPO \
        : \
        -np ${NPROC_MAIA} maia $PROPERTIES_FILE --silent

else
    echo "=== Testing Environment ==="

    # Prepare workspace
    python prepare_workspace.py \
        --env "$ENVIRONMENT" \
        --work-dir "$WORK_DIR"

    cd "$WORK_DIR" || exit 1

    echo "Work directory: $(pwd)"
    echo "Properties file: $PROPERTIES_FILE"
    echo ""

    mpirun \
        -np 1 python ../test_maia_env.py \
            --environment "$ENVIRONMENT" \
            --num-steps ${NUM_STEPS} \
        : \
        -np ${NPROC_MAIA} maia $PROPERTIES_FILE --silent
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
