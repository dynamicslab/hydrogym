#!/usr/bin/env bash

export OMP_NUM_THREADS=1

# load maia solver (here GPU version) -> CPU: MAIA/1.0-gompi-2024a-SystemCUDA-CPU
module load MAIA/1.0-NVHPC-24.11-SystemCUDA-GPU

# Configuration
WORK_DIR="./test_run_005"
ENVIRONMENT="SquareCylinder_2D_Re1000"
PROPERTIES_FILE="properties_run.toml"

# Prepare environment
python prepare_workspace.py --env $ENVIRONMENT --work-dir $WORK_DIR

# Change to work directory
cd "$WORK_DIR" || exit 1

echo "=== MAIA MPMD Coupling ==="
echo "Work directory: $(pwd)"
echo "Environment: $ENVIRONMENT"
echo "Properties file: $PROPERTIES_FILE"
echo ""

# Run M-AIA based hydrogym environment in MPMD mode
mpirun -np 1 python ../test_maia_env.py --environment=$ENVIRONMENT : -np 1 maia $PROPERTIES_FILE --silent
