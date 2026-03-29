#!/usr/bin/env bash

export OMP_NUM_THREADS=1
module load MAIA/1.0-gompi-2024a-SystemCUDA-CPU HDF5/1.14.5-gompi-2024a-SystemCUDA OpenBLAS/0.3.27-gompi-2024a-SystemCUDA Bison/3.8.2-GCCcore-13.3.0 CMake/3.30.3-GCCcore-13.3.0 Cython/3.0.10-GCCcore-13.3.0

# Configuration
ENVIRONMENT="cylinder"

echo "=== Firedrake run ==="
echo "Environment: $ENVIRONMENT"
echo ""

# Run Firedrake based environment
python test_firedrake_env.py --environment=$ENVIRONMENT