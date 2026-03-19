#!/usr/bin/env bash
#
# Run zero-shot wing deployment demo with MPMD coupling.
#
# Usage:
#     ./run_pettingzoo_docker.sh
#     ./run_pettingzoo_docker.sh --policy-root /workspace/legacy_runs

set -e

# Load Nek5000 module
module purge
module load Nek5000/1.0-gompi-2024a-SystemCUDA-SmallWing

# Activate Python environment
source ~/venvs/hydrogym_cpu/bin/activate

export OMP_NUM_THREADS=1

# Configuration
WORK_DIR="./train_run"
LOCAL_DIR="/workspace/hydrogym/packaged_envs"
ENV_NAME="NACA4412_3D_Re75000_AOA5"
NPROC_NEK=12
NUM_STEPS=30
POLICY_TEMPLATE="../meta_policy_small_wing_template.py"
POLICY_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-root)
            POLICY_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=== Zero-Shot Wing Deployment (PettingZoo) ==="
echo "Environment: $ENV_NAME"
echo "Nek5000 procs: $NPROC_NEK"
echo "Rollout steps: $NUM_STEPS"
if [ -n "$POLICY_ROOT" ]; then
    echo "Policy root: $POLICY_ROOT"
else
    echo "Policy root: <template default>"
fi
echo ""

# Prepare workspace
python ../prepare_workspace.py \
    --local-dir "$LOCAL_DIR" \
    --env "$ENV_NAME" \
    --work-dir "$WORK_DIR" \
    --cache-dir "$HOME/.cache/hydrogym" \
    --restart-index 1 \
    --profile NEK5000_v17

cd "$WORK_DIR" || exit 1
# Remove the "sch" file in the run folder
rm -f *.sch

if [ -n "$POLICY_ROOT" ]; then
    mpirun \
        -np 1 python ../zero_shot_demo_pettingzoo.py \
            --env "$ENV_NAME" \
            --nproc ${NPROC_NEK} \
            --steps ${NUM_STEPS} \
            --local-dir "$LOCAL_DIR" \
            --policy-template "$POLICY_TEMPLATE" \
            --policy-root "$POLICY_ROOT" \
        : \
        -np ${NPROC_NEK} nek5000
else
    mpirun \
        -np 1 python ../zero_shot_demo_pettingzoo.py \
            --env "$ENV_NAME" \
            --nproc ${NPROC_NEK} \
            --steps ${NUM_STEPS} \
            --local-dir "$LOCAL_DIR" \
            --policy-template "$POLICY_TEMPLATE" \
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
