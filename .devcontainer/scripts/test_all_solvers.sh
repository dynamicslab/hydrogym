#!/usr/bin/env bash
# Actually RUN each solver's "getting started" example against the
# containers built by build_all_containers.sh - real env.reset()/env.step()
# (or solver-equivalent) calls through the real venvs and solver binaries,
# not just `import hydrogym`. Run build_all_containers.sh first (at least
# the full-cpu-stack and full-gpu-stack configs, which is all this script
# needs); containers are found by the devcontainer.config_variant label it
# leaves behind, not by hardcoded container IDs.
#
# Usage:
#   ./test_all_solvers.sh                              # all 7 solver tests
#   TESTS="maia_gpu firedrake jax_kolmogorov" ./test_all_solvers.sh   # subset
#
# IMPORTANT - internet/Hugging Face Hub dependency:
#   maia_gpu, maia_cpu, firedrake, nek5000, and jax_channel all download
#   environment/checkpoint data from the dynamicslab/HydroGym-environments
#   HF Hub repo on first run (this is how the example scripts are written -
#   see each example's README; none of the "getting started" test scripts
#   expose a CLI flag to force fully offline use). Only jax_kolmogorov is
#   guaranteed local-only. If this host/container has no internet access,
#   expect those five to fail at the download step, not because the solver
#   itself is broken - check the log before concluding a solver regressed.
#
#   nek5000 downloads a packaged env from HF Hub - as of this writing there
#   is no local/offline packaged env anywhere in this repo to fall back to
#   (third_party/nek5000/cases/ has raw build trees, not the packaged format
#   NekDataManager expects), so this test has no offline path at all.
#
# Known upstream doc/example drift worked around below (found by actually
# running these, not by inspection - see conversation history):
#   - test_nek_direct.py's own --env default is "MiniChannel_Re180", which
#     does not exist on the dynamicslab/HydroGym-environments HF repo at
#     all (confirmed via `HfApi().list_repo_files`); the real env for this
#     case is named "TCFmini_3D_Re180" (matches the nek README's own
#     numbered examples, just not this script's default).
#   - examples/maia/getting_started/README.md's Quick Start says the MAIA
#     properties file is symlinked as "properties.toml" in the prepared
#     workspace; it's actually named "properties_run.toml".
#   - maia-cpu's binary is GNU-toolchain-linked and must be launched via the
#     apt-OpenMPI wrapper `mpirun.openmpi`, not bare `mpirun` (which
#     resolves to HPCX's on PATH, an ABI-incompatible libmpi - same trap as
#     the LD_LIBRARY_PATH override just below, just for the mpirun binary
#     itself rather than the solver binary).
#
# jaxfluids's test script has no --num-steps flag (1000 steps hardcoded) -
# every test here (not just this one) runs under a uniform TIMEOUT_SECS cap
# (see below), so this one will almost always hit that cap rather than
# finish; a TIMEOUT is reported separately from a FAIL for exactly this
# reason - check the log for actual step output before assuming it's broken.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/.devcontainer/test-logs"
mkdir -p "${LOG_DIR}"

container_for() {
  # Prints the first running container ID labeled devcontainer.config_variant=$1
  docker ps --filter "label=devcontainer.config_variant=$1" --format '{{.ID}}' | head -1
}

CID_GPU="$(container_for full-gpu-stack)"
CID_CPU="$(container_for full-cpu-stack)"

if [[ -z "${CID_GPU}" ]]; then
  echo "WARNING: no running container labeled devcontainer.config_variant=full-gpu-stack"
  echo "         (maia_gpu, jax_kolmogorov, jax_channel, jaxfluids will be SKIPPED)"
  echo "         Run: ${SCRIPT_DIR}/build_all_containers.sh   (or just the full-gpu-stack config)"
fi
if [[ -z "${CID_CPU}" ]]; then
  echo "WARNING: no running container labeled devcontainer.config_variant=full-cpu-stack"
  echo "         (maia_cpu, firedrake, nek5000 will be SKIPPED)"
  echo "         Run: ${SCRIPT_DIR}/build_all_containers.sh   (or just the full-cpu-stack config)"
fi

declare -A CID=(
  [maia_gpu]="${CID_GPU}"
  [maia_cpu]="${CID_CPU}"
  [firedrake]="${CID_CPU}"
  [nek5000]="${CID_CPU}"
  [jax_kolmogorov]="${CID_GPU}"
  [jax_channel]="${CID_GPU}"
  [jaxfluids]="${CID_GPU}"
)

declare -A CMD

CMD[maia_gpu]='
set -e
source /opt/venvs/maia-gpu/bin/activate
export OMP_NUM_THREADS=1
cd /workspace/hydrogym/examples/maia/getting_started
rm -rf ci_test_run_gpu
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./ci_test_run_gpu
cd ci_test_run_gpu
mpirun --allow-run-as-root -np 1 python ../test_maia_env.py \
    --environment Cylinder_2D_Re200 --num-steps 3 \
    : -np 1 /workspace/wipmaiaml/build_nvhpc_production/bin/maia properties_run.toml --silent
'

CMD[maia_cpu]='
set -e
source /opt/venvs/maia-cpu/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/parallel-netcdf-1.14.0/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}"
cd /workspace/hydrogym/examples/maia/getting_started
rm -rf ci_test_run_cpu
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./ci_test_run_cpu
cd ci_test_run_cpu
mpirun.openmpi --allow-run-as-root -np 1 python ../test_maia_env.py \
    --environment Cylinder_2D_Re200 --num-steps 3 \
    : -np 1 /workspace/wipmaiaml/build_gnu_production/bin/maia properties_run.toml --silent
'

CMD[firedrake]='
set -e
source /opt/venvs/firedrake/activate_firedrake.sh
cd /workspace/hydrogym/examples/firedrake/getting_started
python test_firedrake_env.py --environment cylinder --num-steps 3 --mesh-resolution medium
'

CMD[nek5000]='
set -e
source /opt/venvs/nek5000/bin/activate
export OMP_NUM_THREADS=1
# Best-effort PATH fallback so a bare `nek5000` resolves to a real binary if
# the downloaded packaged env does not carry its own - see header comment,
# this may legitimately not match TCFmini_3D_Re180 and fail for that reason.
export PATH="/workspace/hydrogym/third_party/nek5000/cases/mini_channel:${PATH}"
cd /workspace/hydrogym/examples/nek/getting_started/1_nekenv_single
rm -rf ci_test_run
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir ./ci_test_run --cache-dir "$HOME/.cache/hydrogym"
cd ci_test_run
# nproc/-np MUST be 10, matching how the mini_channel/phill case binary is
# actually compiled (SIZE: lelt=13, lp=10, totctrl=220 - all sized for
# exactly 10 Nek ranks) and how the published TCFmini_3D_Re180
# environment_config.yaml on HF Hub declares TOTCTRL=220. A smaller -np
# looks like it should just mean fewer/faster ranks but actually breaks
# multiple compile-time size assumptions at once (lelt too small for the
# per-rank element count, a single rank owning more wall/control points
# than TOTCTRL, and the node-list MPI handshake overrunning the
# receive buffer on the controller side) - see conversation history.
mpirun --allow-run-as-root --use-hwthread-cpus -np 1 python ../test_nek_direct.py \
    --env TCFmini_3D_Re180 --steps 3 --nproc 10 \
    : -np 10 nek5000
'

CMD[jax_kolmogorov]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/hydrogym/examples/jax/getting_started/1_kolmogorov
python test_kolmogorov_env.py no_actuation --num-steps 5
'

CMD[jax_channel]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/hydrogym/examples/jax/getting_started/2_channel
python test_channel_env.py no_actuation --num-steps 2
'

CMD[jaxfluids]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/hydrogym/examples/jaxfluids
python test_jaxfluids_env.py
'

DEFAULT_TESTS="maia_gpu maia_cpu firedrake nek5000 jax_kolmogorov jax_channel jaxfluids"
TESTS="${TESTS:-${DEFAULT_TESTS}}"

# This is a SMOKE TEST, not a full run: the goal is just "does the env
# actually reset() and step() through the real solver", not a complete
# episode. Every test is capped at TIMEOUT_SECS regardless of which command
# it runs - this matters because an MPMD pairing (maia/nek) where the
# Python side crashes does NOT automatically kill the paired solver ranks,
# which then hang forever waiting for a handshake that will never come (hit
# this for real with nek5000 - see conversation history). Without a uniform
# timeout, one broken test silently blocks the whole suite indefinitely.
TIMEOUT_SECS="${TIMEOUT_SECS:-90}"

RESULTS=()

for name in ${TESTS}; do
  cid="${CID[${name}]:-}"
  log_file="${LOG_DIR}/${name}.log"

  echo ""
  echo "==================================================================="
  echo "Testing '${name}'  (timeout ${TIMEOUT_SECS}s)"
  echo "Log: ${log_file}"
  echo "==================================================================="

  if [[ -z "${cid}" ]]; then
    echo "SKIPPED: no container available for '${name}'"
    RESULTS+=("${name}: SKIPPED (container not running)")
    continue
  fi

  step_start=$(date +%s)
  docker exec "${cid}" timeout -k 10 "${TIMEOUT_SECS}" bash -c "${CMD[${name}]}" 2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  step_elapsed=$(( $(date +%s) - step_start ))
  timing="[$(( step_elapsed / 60 ))m $(( step_elapsed % 60 ))s]"

  # Regardless of clean exit, check whether the log actually shows the
  # thing we care about: the env stepped through the real solver at least
  # once. This is what answers "does it work", independent of whether the
  # process also tore itself down cleanly afterwards.
  # Deliberately specific, not just "step"/"reward" - those words also show
  # up in unrelated boilerplate (e.g. prepare_workspace.py's own "Next
  # steps:" printout, which previously false-positived this check on a run
  # that had actually crashed before ever reaching the solver).
  if grep -qiE "resetting environment|env[._]step[ =]|sim_step=|episode 1/|reward *[:=] *-?[0-9]" "${log_file}"; then
    evidence="(saw step/reset output)"
  else
    evidence="(no step/reset output seen)"
  fi

  if [[ ${rc} -eq 0 ]]; then
    RESULTS+=("${name}: PASS  ${evidence}  ${timing}")
  elif [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
    RESULTS+=("${name}: TIMEOUT ${evidence}  ${timing}")
  else
    RESULTS+=("${name}: FAIL (exit ${rc}) ${evidence}  ${timing}")
  fi
done

echo ""
echo "==================================================================="
echo "Solver test summary"
echo "==================================================================="
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo ""
echo "Reminder: maia_gpu, maia_cpu, firedrake, nek5000, and jax_channel need"
echo "internet access to Hugging Face Hub on first run - a FAIL there may be"
echo "a network issue, not a solver/container regression. Check the log."

for r in "${RESULTS[@]}"; do
  [[ "${r}" == *": FAIL"* ]] && exit 1
done
exit 0
