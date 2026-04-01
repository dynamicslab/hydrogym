#!/usr/bin/env bash
#
# Run Firedrake single-agent tests and training.
#
# Usage:
#     ./run_example_docker.sh                    # Test only
#     ./run_example_docker.sh train              # Train SB3 agent

set -e

export OMP_NUM_THREADS=1

# Load Firedrake modules
module purge
module load MAIA/1.0-gompi-2024a-SystemCUDA-CPU HDF5/1.14.5-gompi-2024a-SystemCUDA mpi4py/4.0.1-gompi-2024a-SystemCUDA

# Activate Python environment
source /home/easybuild/venvs/hydrogym_cpu/bin/activate

# Configuration
ENVIRONMENT="cylinder"
RE=100
MESH="medium"
NUM_STEPS=100
TOTAL_TIMESTEPS=50000
MODE="${1:-test}"  # test or train

echo "=== Firedrake Single Agent ==="
echo "Mode: $MODE"
echo "Environment: $ENVIRONMENT"
echo "Reynolds number: Re=${RE}"
echo "Mesh resolution: $MESH"
echo ""

if [ "$MODE" == "train" ]; then
    echo "=== Training SB3 Agent ==="

    # Run training (pure Python, no MPMD required)
    python train_sb3_firedrake.py \
        --env "$ENVIRONMENT" \
        --reynolds ${RE} \
        --mesh ${MESH} \
        --algo PPO \
        --total-timesteps ${TOTAL_TIMESTEPS}

else
    echo "=== Testing Environment ==="

    # Run test
    python test_firedrake_env.py \
        --environment "$ENVIRONMENT" \
        --reynolds ${RE} \
        --mesh-resolution ${MESH} \
        --num-steps ${NUM_STEPS}
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE