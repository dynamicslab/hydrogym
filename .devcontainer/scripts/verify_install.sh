#!/bin/bash
# Modular DevContainer Installation Verification Script
# Run after container build to verify all enabled features work correctly

set -euo pipefail

echo "=== DevContainer Installation Verification ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    local cmd="$2"
    echo -n "  [$name] "
    if eval "$cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
    fi
}

warn() {
    local name="$1"
    local cmd="$2"
    echo -n "  [$name] "
    if eval "$cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}WARN (optional)${NC}"
        WARN=$((WARN + 1))
    fi
}

echo "--- Base Environment ---"
check "NVHPC 26.5" "nvc++ --version | grep -q '26.5'"
check "CUDA 13.2" "nvcc --version | grep -q '13.2' || nvc++ --version | grep -q '13.2'"
check "HPCX OpenMPI" "mpirun --version | grep -q 'HPC-X' || mpirun --version | grep -q 'Open MPI'"
check "Python 3.12+" "python3 --version | grep -q '3.12' || python3 --version | grep -q '3.13'"
check "uv installed" "command -v uv"
check "cmake" "command -v cmake"
check "ninja" "command -v ninja"

echo ""
echo "--- Base Libraries ---"
check "Parallel-NetCDF" "test -f /opt/maia_deps/parallel-netcdf-1.14.0/lib/libpnetcdf.so"
check "Parallel HDF5" "test -f /opt/maia_deps/hdf5-1.14.5/lib/libhdf5.so"
check "FFTW3 (MPI+OMP)" "test -f /opt/maia_deps/fftw-3.3.10/lib/libfftw3_mpi.so && test -f /opt/maia_deps/fftw-3.3.10/lib/libfftw3_omp.so"
check "LD_LIBRARY_PATH set" "echo \$LD_LIBRARY_PATH | grep -q 'maia_deps'"

echo ""
echo "--- MAIA GPU Feature ---"
if [[ -f "/workspace/third_party/m-AIA/build_nvhpc_production/bin/maia" ]]; then
    check "MAIA binary" "test -f /workspace/third_party/m-AIA/build_nvhpc_production/bin/maia"
    check "MAIA test binary" "test -f /workspace/third_party/m-AIA/build_nvhpc_production/bin/test_maia"
    warn "MAIA tests pass" "/workspace/third_party/m-AIA/build_nvhpc_production/bin/test_maia"
    check "compile_commands.json" "test -f /workspace/third_party/m-AIA/compile_commands.json"
    check "dev_tools present" "test -f /workspace/third_party/m-AIA/auxiliary/dev_tools/regression_test.py && test -f /workspace/third_party/m-AIA/auxiliary/dev_tools/profile_run.py && test -f /workspace/third_party/m-AIA/auxiliary/dev_tools/report.py"
    check "dev_runs dir" "test -d /workspace/dev_runs/results/regression && test -d /workspace/dev_runs/results/profiling"
    warn "HydroGym[maia] import" "/opt/venvs/maia-gpu/bin/python -c 'import hydrogym, importlib.metadata; print(importlib.metadata.version(\"hydrogym\"))'"
else
    echo -e "  [MAIA GPU] ${YELLOW}NOT BUILT (run postCreateCommand or build manually)${NC}"
    WARN=$((WARN + 1))
fi

echo ""
echo "--- MAIA CPU Feature ---"
if [[ -f "/workspace/third_party/m-AIA/build_gnu_production/bin/maia" ]]; then
    check "MAIA CPU binary" "test -f /workspace/third_party/m-AIA/build_gnu_production/bin/maia"
    warn "HydroGym[maia] import" "/opt/venvs/maia-cpu/bin/python -c 'import hydrogym, importlib.metadata; print(importlib.metadata.version(\"hydrogym\"))'"
else
    echo -e "  [MAIA CPU] ${YELLOW}NOT ENABLED${NC}"
fi

echo ""
echo "--- Nek5000 Feature ---"
NEK_CASES_DIR="/workspace/third_party/nek5000/cases"
if [[ -d "${NEK_CASES_DIR}" ]]; then
    for case in mini_channel large_channel naca0012_200k small_wing; do
        warn "Nek5000 ${case}" "test -f ${NEK_CASES_DIR}/${case}/nek5000"
    done
    warn "HydroGym[nek] import" "/opt/venvs/nek5000/bin/python -c 'import hydrogym, importlib.metadata; print(importlib.metadata.version(\"hydrogym\"))'"
else
    echo -e "  [Nek5000] ${YELLOW}NOT ENABLED (or submodules not initialized - run 'git submodule update --init' on the host)${NC}"
fi

echo ""
echo "--- PETSc Feature ---"
PETSC_ARCH_DIR="/opt/petsc/arch-firedrake-default"
if [[ -d "/opt/petsc" ]]; then
    check "PETSc dir" "test -d /opt/petsc"
    check "PETSc libs" "test -f ${PETSC_ARCH_DIR}/lib/libpetsc.so"
else
    echo -e "  [PETSc] ${YELLOW}NOT ENABLED${NC}"
fi

echo ""
echo "--- Firedrake Feature ---"
if [[ -d "/opt/venvs/firedrake" ]]; then
    check "Firedrake venv" "test -d /opt/venvs/firedrake"
    warn "Firedrake import" "/opt/venvs/firedrake/bin/python -c 'import firedrake; print(firedrake.__version__)'"
    warn "HydroGym CPU import" "/opt/venvs/firedrake/bin/python -c 'import hydrogym, importlib.metadata; print(importlib.metadata.version(\"hydrogym\"))'"
else
    echo -e "  [Firedrake] ${YELLOW}NOT ENABLED${NC}"
fi

echo ""
echo "--- Python ML Feature ---"
if [[ -d "/opt/venvs/ml" ]]; then
    check "ML venv" "test -d /opt/venvs/ml"
    warn "PyTorch CUDA" "/opt/venvs/ml/bin/python -c 'import torch; print(torch.cuda.is_available())'"
    warn "JAX devices" "/opt/venvs/ml/bin/python -c 'import jax; print(jax.devices())'"
    warn "stable-baselines3" "/opt/venvs/ml/bin/python -c 'import stable_baselines3; print(stable_baselines3.__version__)'"
else
    echo -e "  [Python ML] ${YELLOW}NOT ENABLED${NC}"
fi

echo ""
echo "--- HydroGym GPU Feature ---"
if [[ -d "/opt/venvs/ml" ]] && /opt/venvs/ml/bin/python -c 'import hydrogym' 2>/dev/null; then
    check "HydroGym GPU" "/opt/venvs/ml/bin/python -c 'import hydrogym, importlib.metadata; print(importlib.metadata.version(\"hydrogym\"))'"
    warn "JAX-Fluids" "/opt/venvs/ml/bin/python -c 'import jaxfluids; print(jaxfluids.__version__)'"
else
    echo -e "  [HydroGym GPU] ${YELLOW}NOT ENABLED${NC}"
fi

echo ""
echo "=== Summary ==="
echo -e "  ${GREEN}Passed: $PASS${NC}"
echo -e "  ${YELLOW}Warnings: $WARN${NC}"
echo -e "  ${RED}Failed: $FAIL${NC}"

if [[ $FAIL -gt 0 ]]; then
    echo -e "\n${RED}Some required checks failed. Please review the build logs.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All required checks passed!${NC}"
    exit 0
fi