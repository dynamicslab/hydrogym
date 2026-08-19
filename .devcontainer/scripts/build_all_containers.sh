#!/usr/bin/env bash
# Build every devcontainer variant in this repo, one at a time, from a clean
# state - never reattaching to a container left running from a previous run
# (the devcontainer CLI otherwise matches an existing container by
# workspace-folder alone, ignoring which config you asked for - see
# conversation history for how that silently reused the wrong container).
#
# Usage:
#   ./build_all_containers.sh                         # from scratch, all 8 configs
#   NO_CACHE=0 ./build_all_containers.sh               # reuse Docker layer cache (much faster)
#   CONFIGS="full-gpu-stack full-cpu-stack" ./build_all_containers.sh   # subset
#
# "From scratch" = --remove-existing-container (always recreate the
# container) + --build-no-cache by default (genuinely rebuilds every image
# layer - including the ~15-25min from-source parallel-netcdf/HDF5/FFTW
# builds in base/Dockerfile.base, built TWICE for two toolchains - for EVERY
# one of the 8 configs below, since each is an independent image build).
# That is expensive: with NO_CACHE=1 (the default) expect this to take
# several hours total, not minutes. Set NO_CACHE=0 to let Docker reuse
# cached layers across configs (the base image layers are identical across
# all 8 configs, so this is dramatically faster) while still forcing a fresh
# CONTAINER + a fresh postCreateCommand run for each one.
#
# Configs run SEQUENTIALLY, not in parallel: several postCreateCommand steps
# do native -j16 compiles (MAIA, PETSc, Firedrake, Nek5000 cases), and
# running two of those at once risks the OOM documented in the top-level
# CLAUDE.md ("some translation units peak at ~4.2GB each").
#
# Each container is left running afterwards, labeled
# devcontainer.config_variant=<name> - test_all_solvers.sh (in this same
# directory) looks containers up by that label, so run this script first.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"          # .../hydrogym
DEVCONTAINER_DIR="${REPO_ROOT}/.devcontainer"
WORKSPACE_FOLDER="${REPO_ROOT}"
LOG_DIR="${DEVCONTAINER_DIR}/build-logs"
mkdir -p "${LOG_DIR}"

NO_CACHE="${NO_CACHE:-1}"

# name -> devcontainer.json file (relative to .devcontainer/)
declare -A CONFIG_FILES=(
  [default]="devcontainer.json"
  [maia-gpu-test]="maia-gpu-test.devcontainer.json"
  [maia-cpu-test]="maia-cpu-test.devcontainer.json"
  [nek5000-test]="nek5000-test.devcontainer.json"
  [petsc-test]="petsc-test.devcontainer.json"
  [firedrake-test]="firedrake-test.devcontainer.json"
  [full-cpu-stack]="full-cpu-stack.devcontainer.json"
  [full-gpu-stack]="full-gpu-stack.devcontainer.json"
)

# Order matters only for how the logs read, not for correctness: each config
# is an independent image/container build. full-cpu-stack and full-gpu-stack
# are last since they duplicate work already done individually above them
# when NO_CACHE=0 (fast); with NO_CACHE=1 they just take as long as everything
# else, independently.
DEFAULT_ORDER="default maia-gpu-test maia-cpu-test nek5000-test petsc-test firedrake-test full-cpu-stack full-gpu-stack"
CONFIGS="${CONFIGS:-${DEFAULT_ORDER}}"

echo "=== build_all_containers.sh ==="
echo "Workspace folder : ${WORKSPACE_FOLDER}"
echo "Configs          : ${CONFIGS}"
echo "NO_CACHE         : ${NO_CACHE} $( [[ "${NO_CACHE}" == "1" ]] && echo '(full from-scratch rebuild - this will take hours)' || echo '(reusing Docker layer cache)' )"
echo "Logs             : ${LOG_DIR}/<name>.log"
echo ""

RESULTS=()
START_TS=$(date +%s)

for name in ${CONFIGS}; do
  file="${CONFIG_FILES[${name}]:-}"
  if [[ -z "${file}" ]]; then
    echo "WARNING: unknown config '${name}' - skipping (known: ${!CONFIG_FILES[*]})"
    RESULTS+=("${name}: SKIPPED (unknown config name)")
    continue
  fi
  if [[ ! -f "${DEVCONTAINER_DIR}/${file}" ]]; then
    echo "WARNING: ${DEVCONTAINER_DIR}/${file} not found - skipping '${name}'"
    RESULTS+=("${name}: SKIPPED (config file missing)")
    continue
  fi

  log_file="${LOG_DIR}/${name}.log"
  step_start=$(date +%s)
  echo ""
  echo "==================================================================="
  echo "Building '${name}'  (${file})"
  echo "Log: ${log_file}"
  echo "==================================================================="

  args=(up
    --workspace-folder "${WORKSPACE_FOLDER}"
    --override-config "${DEVCONTAINER_DIR}/${file}"
    --id-label "devcontainer.config_variant=${name}"
    --remove-existing-container
    --log-level info
  )
  [[ "${NO_CACHE}" == "1" ]] && args+=(--build-no-cache)

  if devcontainer "${args[@]}" 2>&1 | tee "${log_file}"; then
    status="PASS"
  else
    status="FAIL (see ${log_file})"
  fi
  step_elapsed=$(( $(date +%s) - step_start ))
  RESULTS+=("${name}: ${status}  [$(( step_elapsed / 60 ))m $(( step_elapsed % 60 ))s]")
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))

echo ""
echo "==================================================================="
echo "Build summary  (total: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s)"
echo "==================================================================="
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo ""
echo "Containers are left running, each labeled devcontainer.config_variant=<name>."
echo "List them with:  docker ps --filter 'label=devcontainer.config_variant'"
echo "Next: ${SCRIPT_DIR}/test_all_solvers.sh"

# Exit non-zero if anything failed, so this is CI-friendly.
for r in "${RESULTS[@]}"; do
  [[ "${r}" == *": FAIL"* ]] && exit 1
done
exit 0
