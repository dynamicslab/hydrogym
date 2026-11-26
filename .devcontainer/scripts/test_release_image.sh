#!/usr/bin/env bash
# Smoke-test a HydroGym release image (build_release_images.sh): every test
# runs in a fresh `docker run --rm` of the image and runs a real solver
# (env.reset()/env.step() through the image's venv, MPI and solver binaries),
# not just imports. Prints a result table and writes one log per test.
#
# Usage:
#   .devcontainer/scripts/test_release_image.sh <image> [test ...]
#   GPU=0 .devcontainer/scripts/test_release_image.sh <gpu-image> imports mpi4py nek5000
#   .devcontainer/scripts/test_release_image.sh --list
#
# Tests (default: all that apply to the image; GPU tests need GPU=1):
#   imports        Python packages, MPI library of mpi4py, Firedrake, maia builds,
#                  no Git LFS pointer files in /opt/hydrogym, /opt/nek5000/cases,
#                  the released policies (policies/load_policy.py --check)
#   mpi4py         mpirun -np 2 python: allreduce
#   firedrake      Firedrake cylinder, 3 actions, FIREDRAKE_NP MPI ranks (default 1; CPU image)
#   firedrake_sb3  as a non-root user without a home directory (docker run --user 1234:1234):
#                  Firedrake venv activated, PPO training of the cylinder with
#                  train_sb3_firedrake.py, 16 steps (Firedrake caches outside the venv; CPU image)
#   nek5000        Nek5000 minimal channel (TCFmini_3D_Re180), 3 actions, 1 Python + 10 Nek5000 ranks
#   maia           m-AIA 2D cylinder Re 200, released PPO policy, 3 + 3 actions, 1 Python + 2 m-AIA ranks
#   cavity         m-AIA 2D open cavity Re 4200, released PPO policy, 1 + 3 actions, 1 Python + 2 m-AIA ranks
#   nekrs          NekRS channel Re_tau 207, opposition control, 3 actions, 1 Python + 2 NekRS ranks
#                  (CUDA: both on the first GPU, --device-id 0)
#                  (first run JIT-compiles the kernels: ~10-15 min; the cache is kept in CACHE_DIR)
#   jax            JAX Kolmogorov flow, 5 steps (GPU image: on the GPU)
#   jaxfluids      JAX-Fluids nozzle (stops at the time limit; PASS if it stepped)
#
# Environment:
#   GPU        1: docker run --gpus all (default for CUDA images), 0: no GPU
#   UCX_TLS    passed into the container if set (e.g. shm,self,tcp)
#   CACHE_DIR  host directory mounted as /root/.cache (Hugging Face downloads,
#              NekRS kernels); default ~/.cache/hydrogym-image-tests
#   TIMEOUT    seconds per test (default 900; nekrs 2400)
#   LOG_DIR    .devcontainer/test-logs
#   PREPARE_ARGS  passed to the m-AIA policy runs (e.g. "--env-dir /envs/Cavity_2D_Re4200")
#   DOCKER_ARGS   extra docker run options (mounts, --user)
#   DOCKER     docker

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER="${DOCKER:-docker}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/.devcontainer/test-logs}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/hydrogym-image-tests}"
ALL_TESTS="imports mpi4py firedrake firedrake_sb3 nek5000 maia cavity nekrs jax jaxfluids"

if [[ "${1:-}" == "--list" || $# -lt 1 ]]; then
    sed -n '2,36p' "$0"
    exit 0
fi
IMAGE="$1"; shift
TESTS="${*:-${ALL_TESTS}}"
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

cuda_image=0
"${DOCKER}" run --rm --entrypoint "" "${IMAGE}" test -x /opt/maia/libexec/gpu_cc && cuda_image=1
GPU="${GPU:-${cuda_image}}"

declare -A CMD
CMD[imports]='
cat /opt/hydrogym/IMAGE_VERSION
python - <<"EOF"
import importlib.metadata as m
from mpi4py import MPI
import hydrogym, jax, jaxfluids, torch, gymnasium, stable_baselines3
print("hydrogym", m.version("hydrogym"), "| jax", jax.__version__, jax.devices(), "| torch", torch.__version__)
print("mpi4py:", MPI.Get_library_version().splitlines()[0])
EOF
[ ! -x /opt/venvs/firedrake/bin/python ] || /opt/venvs/firedrake/bin/python -c "import firedrake, hydrogym, importlib.metadata as m; print(\"firedrake\", m.version(\"firedrake\"), \"hydrogym\", m.version(\"hydrogym\"))"
if [ -x /opt/maia/libexec/gpu_cc ]; then echo "maia builds: $(maia --list-builds | tr "\n" " ")"; fi
nekrs --help > /dev/null 2>&1 || true
ls /opt/nekrs/bin/nekrs /opt/nekrs/bin/nekrs-mpmd /opt/nek5000/cases/*/nek5000
# Git LFS files (meshes, Nek5000 case files) must be the data, not LFS pointer files
lfs="$(grep -rlI "^version https://git-lfs" /opt/hydrogym /opt/nek5000/cases /opt/venvs/*/lib/python3*/site-packages/hydrogym* 2>/dev/null || true)"
if [ -n "$lfs" ]; then echo "ERROR: Git LFS pointer files instead of the data:"; echo "$lfs"; exit 1; fi
echo "no Git LFS pointer files"
# released policies: every file present (sha256) and a forward pass per entry
python /opt/hydrogym/policies/load_policy.py --check | tail -1 | grep -E "^check: OK \([0-9]+ entries\)"
'
CMD[mpi4py]='
mpirun -np 2 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; s = c.allreduce(c.rank + 1); print(\"rank\", c.rank, \"of\", c.size, \"sum\", s); assert s == 3"
'
CMD[firedrake]='
. /opt/venvs/firedrake/bin/activate
cd /opt/hydrogym/examples/firedrake/getting_started
mpirun -np ${FIREDRAKE_NP:-1} python test_firedrake_env.py --environment cylinder --num-steps 3 --mesh-resolution medium
'
CMD[firedrake_sb3]='
id; . /opt/venvs/firedrake/bin/activate
cd /tmp
python -c "import os, firedrake, stable_baselines3, torch; print(\"stable-baselines3\", stable_baselines3.__version__, \"torch\", torch.__version__); print({k: os.environ.get(k) for k in (\"PYOP2_CACHE_DIR\", \"FIREDRAKE_TSFC_KERNEL_CACHE_DIR\")})"
python /opt/hydrogym/examples/firedrake/getting_started/train_sb3_firedrake.py --env cylinder --mesh medium \
    --algo PPO --total-timesteps 16 --n-steps 8 --batch-size 8 --log-dir /tmp/sb3_logs
ls /tmp/sb3_logs
'
CMD[nek5000]='
cd /opt/hydrogym/examples/nek/getting_started/1_nekenv_single
rm -rf /tmp/nek && python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir /tmp/nek --cache-dir /root/.cache/hydrogym
cd /tmp/nek
mpirun --bind-to none --oversubscribe -np 1 python /opt/hydrogym/examples/nek/getting_started/1_nekenv_single/test_nek_direct.py \
    --env TCFmini_3D_Re180 --steps 3 --nproc 10 : -np 10 /opt/nek5000/cases/mini_channel/nek5000
'
CMD[maia]='
cd /opt/hydrogym/examples/maia/released_policies
MPIRUN="mpirun --bind-to none" PLOT=0 ./run_example.sh cylinder_2d_re200/ppo/seed1 /tmp/maia_cyl 2 --actions 3 --start 3
python -c "import numpy as np; d = np.load(\"/tmp/maia_cyl/run.npz\"); print({k: np.round(d[k], 4).tolist() for k in d.files if k.endswith(\"C_D\")})"
'
CMD[cavity]='
cd /opt/hydrogym/examples/maia/released_policies
MPIRUN="mpirun --bind-to none --oversubscribe" PLOT=0 ./run_example.sh cavity_2d_re4200/ppo_tuned/seed0 /tmp/maia_cav 2 --actions 3 --start 1
python -c "import numpy as np; d = np.load(\"/tmp/maia_cav/run.npz\"); print({k: np.round(d[k], 5).tolist() for k in d.files if k.endswith((\"_O\", \"_action\"))})"
'
CMD[nekrs]='
cd /opt/hydrogym/examples/nekrs/channel
export NEKRS_CACHE_DIR=/root/.cache/nekrs-${NEKRS_BACKEND}
# Both NekRS ranks on the first GPU (default: one GPU per local rank)
printf "#!/bin/sh\nexec nekrs-mpmd \"\$@\" --device-id 0\n" > /tmp/nekrs-mpmd-dev0 && chmod +x /tmp/nekrs-mpmd-dev0
[ "${NEKRS_BACKEND}" = CUDA ] && export NEKRS_MPMD=/tmp/nekrs-mpmd-dev0
NP=2 RUN=/tmp/nekrs_run ./run.sh --policy oppo --steps 3
'
CMD[jax]='
cd /opt/hydrogym/examples/jax/getting_started/1_kolmogorov
python -c "import jax; print(jax.devices())"
python test_kolmogorov_env.py no_actuation --num-steps 5
'
CMD[jaxfluids]='
cd /opt/hydrogym/examples/jaxfluids
python test_jaxfluids_env.py
'

run_args=(--rm --shm-size=8g -v "${CACHE_DIR}:/root/.cache")
[[ "${GPU}" == "1" ]] && run_args+=(--gpus all)
[[ -n "${UCX_TLS:-}" ]] && run_args+=(-e "UCX_TLS=${UCX_TLS}")
[[ -n "${FIREDRAKE_NP:-}" ]] && run_args+=(-e "FIREDRAKE_NP=${FIREDRAKE_NP}")
[[ -n "${PREPARE_ARGS:-}" ]] && run_args+=(-e "PREPARE_ARGS=${PREPARE_ARGS}")
# e.g. DOCKER_ARGS="-v /data/hf:/envs:ro --user 1000:1000 -e HOME=/tmp"
# shellcheck disable=SC2206
[[ -n "${DOCKER_ARGS:-}" ]] && run_args+=(${DOCKER_ARGS})
tag="$(echo "${IMAGE##*/}" | tr ':/' '__')_gpu${GPU}${UCX_TLS:+_ucx-${UCX_TLS//,/-}}"

results=()
rc=0
for t in ${TESTS}; do
    if [[ -z "${CMD[$t]:-}" ]]; then echo "unknown test '$t' (${ALL_TESTS})"; exit 1; fi
    if [[ "${GPU}" != "1" && "${cuda_image}" == "1" && " maia cavity nekrs " == *" $t "* ]]; then
        results+=("$(printf '%-13s %-8s %6s  %s' "$t" SKIP - "needs a GPU")"); continue
    fi
    if [[ "$t" == firedrake* ]] && ! "${DOCKER}" run --rm --entrypoint "" "${IMAGE}" test -d /opt/venvs/firedrake; then
        results+=("$(printf '%-13s %-8s %6s  %s' "$t" SKIP - "not in this image")"); continue
    fi
    limit="${TIMEOUT:-900}"; [[ "$t" == nekrs ]] && limit="${TIMEOUT:-2400}"
    [[ "$t" == jaxfluids ]] && limit="${TIMEOUT:-240}"
    log="${LOG_DIR}/${tag}_${t}.log"
    start=$(date +%s)
    name="hgtest-$$-${t}"
    extra=(); [[ "$t" == firedrake_sb3 ]] && extra=(--user 1234:1234)
    timeout --foreground -k 30 "${limit}" "${DOCKER}" run --name "${name}" "${run_args[@]}" "${extra[@]}" "${IMAGE}" \
        bash -c "set -eo pipefail; ${CMD[$t]}" > "${log}" 2>&1
    status=$?
    # the docker client does not stop the container when it is killed
    [[ ${status} -eq 124 || ${status} -eq 137 ]] && { "${DOCKER}" rm -f "${name}" > /dev/null 2>&1; status=124; }
    elapsed=$(( $(date +%s) - start ))
    if [[ ${status} -eq 0 ]]; then
        res=PASS
    elif [[ ${status} -eq 124 && "$t" == jaxfluids ]] && grep -q -E "env_step=[1-9]" "${log}"; then
        res=PASS
    elif [[ ${status} -eq 124 ]]; then
        res=TIMEOUT; rc=1
    else
        res=FAIL; rc=1
    fi
    results+=("$(printf '%-13s %-8s %5ss  %s' "$t" "${res}" "${elapsed}" "${log}")")
    echo "${results[-1]}"
done

echo ""
echo "=== ${IMAGE}  GPU=${GPU}${UCX_TLS:+  UCX_TLS=${UCX_TLS}} ==="
printf '%s\n' "${results[@]}"
exit "${rc}"
