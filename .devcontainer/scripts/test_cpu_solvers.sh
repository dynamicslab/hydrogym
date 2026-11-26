#!/usr/bin/env bash
# Actually RUN each CPU-stack solver's "getting started" example against the
# full-cpu-stack container built by build_all_containers.sh - real
# env.reset()/env.step() calls through the real venv and solver binary, not
# just `import hydrogym`. The GPU and CPU stacks are tested independently (e.g. on a node with no GPU, or
# without waiting on the other stack's build) - see test_gpu_solvers.sh for
# maia_gpu/jax_kolmogorov/jax_channel/jaxfluids.
#
# Usage:
#   ./test_cpu_solvers.sh                              # all CPU-stack tests
#   TESTS="maia_cpu nek5000" ./test_cpu_solvers.sh      # subset
#
# IMPORTANT - internet/Hugging Face Hub dependency:
#   maia_cpu, firedrake, and nek5000 all download environment/checkpoint
#   data from the dynamicslab/HydroGym-environments HF Hub repo on first run
#   (this is how the example scripts are written - see each example's
#   README; none of the "getting started" test scripts expose a CLI flag to
#   force fully offline use). If this host/container has no internet
#   access, expect these to fail at the download step, not because the
#   solver itself is broken - check the log before concluding a solver
#   regressed.
#
#   nek5000 downloads a packaged env from HF Hub - there is no
#   local/offline packaged env in this repo to fall back to
#   (third_party/nek5000/cases/ has raw build trees, not the packaged format
#   HFDataManager expects), so this test has no offline path at all.
#
# Notes on the commands below:
#   - maia-cpu's binary is GNU-toolchain-linked and must be launched via the
#     apt-OpenMPI wrapper `mpirun.openmpi`, not bare `mpirun` (which
#     resolves to HPCX's on PATH, an ABI-incompatible libmpi - same trap as
#     the LD_LIBRARY_PATH override just below, just for the mpirun binary
#     itself rather than the solver binary).
#   - nek5000's small_wing/NACA4412_3D_Re75000_AOA5 env is packaged in the
#     older NEK5000_v17 (.rea) format, not v19 (.par) like TCFmini - needs
#     `prepare_workspace.py --profile NEK5000_v17` or it can't find the
#     case file at all.
#   - that env's config has no normalization.dUdy/runner.dUdy (its reward
#     varies with streamwise position, not dU/dy), so the test passes
#     --baseline-dudy (zeroshot_demo_pettingzoo.py sets env.baseline_dudy
#     after construction); without it NekEnv.step() stops with an error.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/.devcontainer/test-logs"
mkdir -p "${LOG_DIR}"

# shellcheck source=lib_solver_test_runner.sh
source "${SCRIPT_DIR}/lib_solver_test_runner.sh"

CID_CPU="$(container_for full-cpu-stack)"
if [[ -z "${CID_CPU}" ]]; then
  echo "WARNING: no running container labeled devcontainer.config_variant=full-cpu-stack"
  echo "         Run: ${SCRIPT_DIR}/build_all_containers.sh   (or just the full-cpu-stack config)"
fi

declare -A CMD

CMD[maia_cpu]='
set -e
source /opt/venvs/maia-cpu/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/opt/maia_deps_gnu/hdf5-1.14.5/lib:/opt/maia_deps_gnu/parallel-netcdf-1.14.0/lib:/opt/maia_deps_gnu/fftw-3.3.10/lib:${LD_LIBRARY_PATH:-}"
cd /workspace/examples/maia/getting_started
rm -rf ci_test_run_cpu
python prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./ci_test_run_cpu
cd ci_test_run_cpu
mpirun.openmpi --allow-run-as-root -np 1 python ../test_maia_env.py \
    --environment Cylinder_2D_Re200 --num-steps 3 \
    : -np 1 /workspace/third_party/m-AIA/build_gnu_production/bin/maia properties_run.toml --silent
'

CMD[firedrake]='
set -e
source /opt/venvs/firedrake/activate_firedrake.sh
cd /workspace/examples/firedrake/getting_started
python test_firedrake_env.py --environment cylinder --num-steps 3 --mesh-resolution medium
'

CMD[nek5000]='
set -e
source /opt/venvs/nek5000/bin/activate
export OMP_NUM_THREADS=1
# Best-effort PATH fallback so a bare `nek5000` resolves to a real binary if
# the downloaded packaged env does not carry its own - see header comment,
# this may legitimately not match TCFmini_3D_Re180 and fail for that reason.
export PATH="/workspace/third_party/nek5000/cases/mini_channel:${PATH}"
cd /workspace/examples/nek/getting_started/1_nekenv_single
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
# receive buffer on the controller side).
mpirun --allow-run-as-root --use-hwthread-cpus -np 1 python ../test_nek_direct.py \
    --env TCFmini_3D_Re180 --steps 3 --nproc 10 \
    : -np 10 nek5000
'

CMD[nek5000_wing]='
set -e
source /opt/venvs/nek5000/bin/activate
export OMP_NUM_THREADS=1
export PATH="/workspace/third_party/nek5000/cases/small_wing:${PATH}"
cd /workspace/examples/nek/getting_started/1_nekenv_single
rm -rf ci_test_wing
python ../prepare_workspace.py --env NACA4412_3D_Re75000_AOA5 --work-dir ./ci_test_wing --cache-dir "$HOME/.cache/hydrogym" --profile NEK5000_v17
cd ci_test_wing
# -np MUST be 12, matching small_wing/SIZE (lp=12, lelt=875) and the
# published env'\''s environment_config.yaml (nproc=12, TOTCTRL=1000) - same
# class of compile-time-sizing trap as mini_channel'\''s -np 10 above.
mpirun --allow-run-as-root --use-hwthread-cpus -np 1 python ../test_nek_direct.py \
    --env NACA4412_3D_Re75000_AOA5 --steps 3 --nproc 12 --baseline-dudy 1.0 \
    : -np 12 nek5000
'

DEFAULT_TESTS="maia_cpu firedrake nek5000 nek5000_wing"
TESTS="${TESTS:-${DEFAULT_TESTS}}"
TIMEOUT_SECS="${TIMEOUT_SECS:-90}"

run_solver_tests "${CID_CPU}"
rc=$?

echo ""
echo "Reminder: maia_cpu, firedrake, and nek5000 need internet access to"
echo "Hugging Face Hub on first run - a FAIL there may be a network issue,"
echo "not a solver/container regression. Check the log."

exit "${rc}"
