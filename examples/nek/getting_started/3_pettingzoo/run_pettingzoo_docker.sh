#!/usr/bin/env bash
#
# Run NEK5000 PettingZoo tests with MPMD coupling.
#
# Usage:
#     ./run_pettingzoo_docker.sh                    # Test only
#     ./run_pettingzoo_docker.sh train              # Train SB3 agent
#     ./run_pettingzoo_docker.sh train ./configs/pettingzoo_tcfmini_re180.yml

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
TOTAL_TIMESTEPS=50000
MODE="${1:-test}"  # test or train
CONFIG_FILE="${2:-pettingzoo_tcfmini_re180.yml}"
# Convert CONFIG_FILE to absolute path
CONFIG_FILE=$(realpath "../../configs/$CONFIG_FILE")

echo "=== NEK5000 PettingZoo (SuperSuit) ==="
echo "Mode: $MODE"
echo "Environment: $ENV_NAME"
echo "Nek5000 procs: $NPROC_NEK"
echo "Config file: $CONFIG_FILE"
echo ""

if [ "$MODE" == "train" ]; then
    echo "=== Training SB3 Agent (SuperSuit Production Wrapper) ==="

    # Prepare workspace
    python ../prepare_workspace.py \
        --local-dir "$LOCAL_DIR" \
        --env "$ENV_NAME" \
        --work-dir "$WORK_DIR" \
        --cache-dir "$HOME/.cache/hydrogym"

    cd "$WORK_DIR" || exit 1

    mpirun \
        -np 1 python ../train_sb3_pettingzoo.py \
            --env "$ENV_NAME" \
            --local-dir "$LOCAL_DIR" \
            --nproc ${NPROC_NEK} \
            --config-file "$CONFIG_FILE" \
            --total-timesteps ${TOTAL_TIMESTEPS} \
            --algo PPO \
        : \
        -np ${NPROC_NEK} bash -c "nek5000 > /dev/null 2>&1"

else
    echo "=== Testing PettingZoo Interface ==="

    # Prepare workspace
    python ../prepare_workspace.py \
        --local-dir "$LOCAL_DIR" \
        --env "$ENV_NAME" \
        --work-dir "$WORK_DIR" \
        --cache-dir "$HOME/.cache/hydrogym"

    cd "$WORK_DIR" || exit 1

    mpirun \
        -np 1 python ../test_nek_pettingzoo.py \
            --env "$ENV_NAME" \
            --local-dir "$LOCAL_DIR" \
            --steps ${NUM_STEPS} \
            --nproc ${NPROC_NEK} \
            --config-file "$CONFIG_FILE" \
        : \
        -np ${NPROC_NEK} nek5000
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully!"
else
    echo "✗ Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
