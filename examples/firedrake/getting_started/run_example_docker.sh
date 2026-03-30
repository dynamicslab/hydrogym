#!/usr/bin/env bash

export OMP_NUM_THREADS=1
module load MAIA/1.0-gompi-2024a-SystemCUDA-CPU HDF5/1.14.5-gompi-2024a-SystemCUDA mpi4py/4.0.1-gompi-2024a-SystemCUDA

# Configuration
ENVIRONMENT="cavity"
RE=7500
MESH="fine"

echo "=== Firedrake run ==="
echo "Environment: $ENVIRONMENT"
echo "Reynolds number: Re=${RE}"
echo "Mesh resolution: $MESH"

# Run Firedrake based environment
python test_firedrake_env.py --environment=$ENVIRONMENT --reynolds=${RE} --mesh-resolution=${MESH}