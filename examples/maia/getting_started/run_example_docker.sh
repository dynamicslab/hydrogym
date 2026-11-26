#!/usr/bin/env bash
#
# Run MAIA single-agent tests and training with MPMD coupling.
#
# A plain MPMD launch script (it does not start a container): it runs as is in the HydroGym container images
# and on a cluster node (from an sbatch/srun script) with HydroGym and m-AIA
# installed.
#
# Usage:
#     ./run_example_docker.sh                    # Test only
#     ./run_example_docker.sh train              # Train SB3 agent

set -e

# Python environment: the HydroGym container images have it in /opt/venvs;
# elsewhere, activate your own environment before running this script.
if [ -f /opt/venvs/hydrogym/bin/activate ]; then source /opt/venvs/hydrogym/bin/activate; fi
# m-AIA executable: `maia` on PATH (in the GPU images a wrapper that starts
# the build for the visible GPU); MAIA=/path/to/maia to use another one.
MAIA="${MAIA:-maia}"

export OMP_NUM_THREADS=1

# Configuration
WORK_DIR="./train_run"
ENVIRONMENT="Cylinder_2D_Re200"
PROPERTIES_FILE="properties_run.toml"
NPROC_MAIA=1   # num GPUs or cpu cores
NUM_STEPS=100
TOTAL_TIMESTEPS=50000
MODE="${1:-test}"  # test or train
EXIT_CODE=0        # exit status of the mpirun below (captured without tripping `set -e`)

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

    # `|| EXIT_CODE=$?` captures a nonzero status without tripping `set -e`,
    # so the reporting block after `fi` always executes.
    mpirun \
        -np 1 python ../train_sb3_maia.py \
            --env "$ENVIRONMENT" \
            --total-timesteps ${TOTAL_TIMESTEPS} \
            --algo PPO \
        : \
        -np ${NPROC_MAIA} "$MAIA" $PROPERTIES_FILE --silent || EXIT_CODE=$?

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
        -np ${NPROC_MAIA} "$MAIA" $PROPERTIES_FILE --silent || EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
