#!/usr/bin/env bash
# Actually RUN each GPU-stack solver's "getting started" example against the
# full-gpu-stack container built by build_all_containers.sh - real
# env.reset()/env.step() calls through the real venv and solver binary, not
# just `import hydrogym`. Split from a single test_all_solvers.sh so the GPU
# and CPU stacks can be tested independently (e.g. on a GPU-only node, or
# without waiting on the other stack's build) - see test_cpu_solvers.sh for
# maia_cpu/firedrake/nek5000.
#
# Usage:
#   ./test_gpu_solvers.sh                                   # all 4 GPU-stack tests
#   TESTS="maia_gpu jax_kolmogorov" ./test_gpu_solvers.sh    # subset
#
# IMPORTANT - internet/Hugging Face Hub dependency:
#   maia_gpu and jax_channel download environment/checkpoint data from the
#   dynamicslab/HydroGym-environments HF Hub repo on first run (this is how
#   the example scripts are written - see each example's README; neither
#   "getting started" test script exposes a CLI flag to force fully offline
#   use). Only jax_kolmogorov is guaranteed local-only. If this
#   host/container has no internet access, expect those two to fail at the
#   download step, not because the solver itself is broken - check the log
#   before concluding a solver regressed.
#
# Known upstream doc/example drift worked around below (found by actually
# running this, not by inspection - see conversation history):
#   - examples/maia/getting_started/README.md's Quick Start says the MAIA
#     properties file is symlinked as "properties.toml" in the prepared
#     workspace; it's actually named "properties_run.toml".
#
# jaxfluids's test script has no --num-steps flag (1000 steps hardcoded) -
# every test here runs under a uniform TIMEOUT_SECS cap (see
# lib_solver_test_runner.sh), so this one will almost always hit that cap
# rather than finish; a TIMEOUT is reported separately from a FAIL for
# exactly this reason - check the log for actual step output before
# assuming it's broken.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/.devcontainer/test-logs"
mkdir -p "${LOG_DIR}"

# shellcheck source=lib_solver_test_runner.sh
source "${SCRIPT_DIR}/lib_solver_test_runner.sh"

CID_GPU="$(container_for full-gpu-stack)"
if [[ -z "${CID_GPU}" ]]; then
  echo "WARNING: no running container labeled devcontainer.config_variant=full-gpu-stack"
  echo "         Run: ${SCRIPT_DIR}/build_all_containers.sh   (or just the full-gpu-stack config)"
fi

declare -A CMD

CMD[maia_gpu]='
set -e
source /opt/venvs/maia-gpu/bin/activate
export OMP_NUM_THREADS=1
cd /workspace/examples/maia/getting_started
rm -rf ci_test_run_gpu
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./ci_test_run_gpu
cd ci_test_run_gpu
mpirun --allow-run-as-root -np 1 python ../test_maia_env.py \
    --environment Cylinder_2D_Re200 --num-steps 3 \
    : -np 1 /workspace/third_party/m-AIA/build_nvhpc_production/bin/maia properties_run.toml --silent
'

CMD[jax_kolmogorov]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/examples/jax/getting_started/1_kolmogorov
python test_kolmogorov_env.py no_actuation --num-steps 5
'

CMD[jax_channel]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/examples/jax/getting_started/2_channel
python test_channel_env.py no_actuation --num-steps 2
'

CMD[jaxfluids]='
set -e
source /opt/venvs/ml/bin/activate
cd /workspace/examples/jaxfluids
python test_jaxfluids_env.py
'

DEFAULT_TESTS="maia_gpu jax_kolmogorov jax_channel jaxfluids"
TESTS="${TESTS:-${DEFAULT_TESTS}}"
TIMEOUT_SECS="${TIMEOUT_SECS:-90}"

run_solver_tests "${CID_GPU}"
rc=$?

echo ""
echo "Reminder: maia_gpu and jax_channel need internet access to Hugging"
echo "Face Hub on first run - a FAIL there may be a network issue, not a"
echo "solver/container regression. Check the log."

exit "${rc}"
