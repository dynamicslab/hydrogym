#!/usr/bin/env bash
# Smoke-test the HydroGym Apptainer image for GH200 nodes
# (build_apptainer_arm64.sh) on one node: every test runs real solvers
# (env.reset()/env.step() through the image's venv, MPI and solver binaries),
# in a fresh `apptainer exec --nv --cleanenv` of the image, with HOME (and so
# ~/.cache: Hugging Face downloads, NekRS kernels) in RUN_DIR/home. Prints a
# result table and writes one log per test.
#
# Usage (GPU tests inside a Slurm allocation of one GH200 node, 4 GPUs):
#   .devcontainer/scripts/test_apptainer_image.sh <image.sif> [test ...]
#   .devcontainer/scripts/test_apptainer_image.sh <image.sif> prepare     # once, on a node with internet
#   .devcontainer/scripts/test_apptainer_image.sh --list
#
# Tests (default: all but prepare):
#   prepare    download the case data of the tests below into RUN_DIR/home/.cache
#              (compute nodes often have no internet; the tests then run with HF_HUB_OFFLINE=1)
#   imports    Python packages, MPI library of mpi4py, JAX devices, m-AIA builds,
#              no Git LFS pointer files in /opt/hydrogym, /opt/nek5000/cases,
#              the released policies (policies/load_policy.py --check)
#   mpi_cpu    container mpirun -np 4 python: allreduce of host memory
#   mpi_gpu    container mpirun -np 4 python: allreduce of GPU memory (CUDA-aware MPI), one GPU per rank
#   srun_cpu   srun --mpi=${SRUN_MPI} -n 4 apptainer exec ...: allreduce of host memory
#   srun_gpu   the same with GPU memory
#   maia       m-AIA 2D cylinder Re 200, released PPO policy, 3 + 3 actions, 1 Python + 2 m-AIA ranks (container mpirun)
#   maia_srun  the same MPMD launch with srun --multi-prog, one apptainer exec per rank, m-AIA on GPUs 0 and 1
#   nekrs      NekRS channel Re_tau 207, opposition control, 3 actions, 1 Python + NEKRS_NP NekRS ranks (one GPU each)
#              (the first run JIT-compiles the kernels; the cache is kept in RUN_DIR/home/.cache)
#   nek5000    Nek5000 minimal channel (TCFmini_3D_Re180), 3 actions, 1 Python + 10 Nek5000 ranks
#   jax        JAX Kolmogorov flow, 5 steps on the GPU
#   jaxfluids  JAX-Fluids nozzle (stops at the time limit; PASS if it stepped)
#
# Environment:
#   RUN_DIR    writable directory for the runs and caches (default ./hydrogym-image-tests)
#   LOG_DIR    RUN_DIR/logs
#   SRUN_MPI   srun --mpi plugin for the srun tests (default: from `srun --mpi=list`, pmix if listed, else
#              pspmix (ParaStation), else none)
#   NEKRS_NP   NekRS ranks (default 2)
#   TIMEOUT    seconds per test (default 900; nekrs 2400)
#   APPTAINER  apptainer

set -uo pipefail

APPTAINER="${APPTAINER:-apptainer}"
ALL_TESTS="imports mpi_cpu mpi_gpu srun_cpu srun_gpu maia maia_srun nekrs nek5000 jax jaxfluids"
if [[ "${1:-}" == "--list" || $# -lt 1 ]]; then
    sed -n '2,38p' "$0"
    exit 0
fi
SIF="$(realpath "$1")"; shift
TESTS="${*:-${ALL_TESTS}}"
RUN_DIR="$(realpath -m "${RUN_DIR:-${PWD}/hydrogym-image-tests}")"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
# srun --mpi plugin: pmix if this Slurm has it, else ParaStation's pspmix, else none
default_srun_mpi() {
    local plugins
    plugins="$(srun --mpi=list 2>&1 | awk 'NF == 1 {print $1}')" || true
    for p in pmix pspmix; do
        if grep -qx "${p}" <<< "${plugins}"; then echo "${p}"; return; fi
    done
    echo none
}
if [[ -z "${SRUN_MPI:-}" ]]; then
    if command -v srun > /dev/null; then SRUN_MPI="$(default_srun_mpi)"; else SRUN_MPI=none; fi
fi
NEKRS_NP="${NEKRS_NP:-2}"
mkdir -p "${RUN_DIR}/home" "${RUN_DIR}/runs" "${LOG_DIR}"

# The NekRS channel case writes into its own directory (/opt/hydrogym/examples/nekrs/channel,
# read-only in the image): a writable copy is bound over it (its udf includes HydroGym's
# header by a relative path, so it has to stay at that path).
CHANNEL="${RUN_DIR}/nekrs_channel"
if [[ ! -d "${CHANNEL}" ]]; then
    rm -rf "${CHANNEL}.part" && mkdir -p "${CHANNEL}.part" && \
        "${APPTAINER}" exec "${SIF}" cp -a /opt/hydrogym/examples/nekrs/channel/. "${CHANNEL}.part/" && \
        mv "${CHANNEL}.part" "${CHANNEL}"
fi

offline=1
[[ " ${TESTS} " == *" prepare "* ]] && offline=0
APPTAINER_ARGS=(--nv --cleanenv --home "${RUN_DIR}/home" --bind "${RUN_DIR}"
    --bind "${CHANNEL}:/opt/hydrogym/examples/nekrs/channel"
    --env "HG_RUNS=${RUN_DIR}/runs,HF_HUB_OFFLINE=${offline},PYTHONUNBUFFERED=1")
[[ -n "${UCX_TLS:-}" ]] && APPTAINER_ARGS+=(--env "UCX_TLS=${UCX_TLS}")
run_in() { "${APPTAINER}" exec "${APPTAINER_ARGS[@]}" "${SIF}" bash -c "set -eo pipefail; $1"; }

ALLREDUCE_CPU='
from mpi4py import MPI
import numpy as np
c = MPI.COMM_WORLD
a = np.full(1 << 20, c.rank + 1.0); b = np.empty_like(a)
c.Allreduce(a, b)
exp = c.size * (c.size + 1) / 2
print("rank", c.rank, "of", c.size, "sum", b[0], "|", MPI.Get_library_version().splitlines()[0])
assert np.all(b == exp)
'
# Device buffers (cudaMalloc) through __cuda_array_interface__ (mpi4py), i.e.
# CUDA-aware MPI; one GPU per rank.
ALLREDUCE_GPU='
import os, ctypes
local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local % 4)
from mpi4py import MPI
import numpy as np
c = MPI.COMM_WORLD
rt = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so.12")
n = 1 << 20
class Dev:
    def __init__(self):
        self.p = ctypes.c_void_p()
        assert rt.cudaMalloc(ctypes.byref(self.p), ctypes.c_size_t(8 * n)) == 0
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<f8", "data": (self.p.value, False), "version": 3, "strides": None, "stream": None}
a, b = Dev(), Dev()
h = np.full(n, c.rank + 1.0)
assert rt.cudaMemcpy(a.p, h.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(8 * n), 1) == 0
c.Allreduce(a, b)
assert rt.cudaMemcpy(h.ctypes.data_as(ctypes.c_void_p), b.p, ctypes.c_size_t(8 * n), 2) == 0
exp = c.size * (c.size + 1) / 2
print("rank", c.rank, "of", c.size, "GPU", os.environ["CUDA_VISIBLE_DEVICES"], "sum", h[0], h[-1])
assert np.all(h == exp)
'

declare -A CMD
CMD[prepare]='
# (copies of files from the read-only image are read-only)
chmod -R u+w $HG_RUNS/nek_prepare $HG_RUNS/maia_prepare 2>/dev/null || true
rm -rf $HG_RUNS/nek_prepare $HG_RUNS/maia_prepare
cd /opt/hydrogym/examples/nek/getting_started/1_nekenv_single
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir $HG_RUNS/nek_prepare --cache-dir $HOME/.cache/hydrogym
# (NekEnv and the JAX-Fluids environments read from the default Hugging Face cache)
python -c "from huggingface_hub import snapshot_download as d; d(\"dynamicslab/HydroGym-environments\", repo_type=\"dataset\", allow_patterns=[\"TCFmini_3D_Re180/*\", \"Nozzle2D_coarse/*\"])"
python /opt/hydrogym/examples/maia/released_policies/prepare_run.py --policy cylinder_2d_re200/ppo/seed1 --work-dir $HG_RUNS/maia_prepare
cd /opt/hydrogym/examples/nekrs/channel
[ -f tcf.re2 ] && [ -f rs6tcf0.f00004 ] || python ../fetch_case.py
'
CMD[imports]='
python - <<"EOF"
import importlib.metadata as m
from mpi4py import MPI
import hydrogym, jax, jaxfluids, torch, gymnasium, stable_baselines3
print("hydrogym", m.version("hydrogym"), "| jax", jax.__version__, jax.devices(), "| torch", torch.__version__)
print("mpi4py:", MPI.Get_library_version().splitlines()[0])
EOF
echo "maia builds: $(maia --list-builds | tr "\n" " ")"
maia --list-builds | grep -q cc90
ls /opt/nekrs/bin/nekrs /opt/nekrs/bin/nekrs-mpmd /opt/nek5000/cases/*/nek5000
cat /opt/hydrogym/IMAGE_VERSION
# Git LFS files (meshes, Nek5000 case files) must be the data, not LFS pointer files
lfs="$(grep -rlI "^version https://git-lfs" /opt/hydrogym /opt/nek5000/cases /opt/venvs/*/lib/python3*/site-packages/hydrogym* 2>/dev/null || true)"
if [ -n "$lfs" ]; then echo "ERROR: Git LFS pointer files instead of the data:"; echo "$lfs"; exit 1; fi
echo "no Git LFS pointer files"
# released policies: every file present (sha256) and a forward pass per entry
python /opt/hydrogym/policies/load_policy.py --check | tail -1 | grep -E "^check: OK \([0-9]+ entries\)"
'
CMD[mpi_cpu]="
mpirun --bind-to none -np 4 python -c '${ALLREDUCE_CPU}'
"
CMD[mpi_gpu]="
mpirun --bind-to none -np 4 python -c '${ALLREDUCE_GPU}'
"
CMD[maia]='
cd /opt/hydrogym/examples/maia/released_policies
rm -rf $HG_RUNS/maia_cyl
MPIRUN="mpirun --bind-to none" PLOT=0 ./run_example.sh cylinder_2d_re200/ppo/seed1 $HG_RUNS/maia_cyl 2 --actions 3 --start 3
python -c "import numpy as np; d = np.load(\"$HG_RUNS/maia_cyl/run.npz\"); print({k: np.round(d[k], 6).tolist() for k in d.files if k.endswith(\"C_D\") or k.endswith(\"C_L\")})"
'
CMD[nekrs]="
cd /opt/hydrogym/examples/nekrs/channel
export NEKRS_CACHE_DIR=\$HOME/.cache/nekrs-\${NEKRS_BACKEND}
rm -rf \$HG_RUNS/nekrs_run
NP=${NEKRS_NP} RUN=\$HG_RUNS/nekrs_run ./run.sh --policy oppo --steps 3
"
CMD[nek5000]='
cd /opt/hydrogym/examples/nek/getting_started/1_nekenv_single
chmod -R u+w $HG_RUNS/nek 2>/dev/null || true
rm -rf $HG_RUNS/nek
python ../prepare_workspace.py --env TCFmini_3D_Re180 --work-dir $HG_RUNS/nek --cache-dir $HOME/.cache/hydrogym
cd $HG_RUNS/nek
mpirun --bind-to none -np 1 python /opt/hydrogym/examples/nek/getting_started/1_nekenv_single/test_nek_direct.py \
    --env TCFmini_3D_Re180 --steps 3 --nproc 10 : -np 10 /opt/nek5000/cases/mini_channel/nek5000
'
CMD[jax]='
mkdir -p $HG_RUNS/jax && cd $HG_RUNS/jax
python -c "import jax; print(jax.devices()); assert jax.default_backend() == \"gpu\""
python /opt/hydrogym/examples/jax/getting_started/1_kolmogorov/test_kolmogorov_env.py no_actuation --num-steps 5
'
CMD[jaxfluids]='
mkdir -p $HG_RUNS/jaxfluids && cd $HG_RUNS/jaxfluids
cp -rn /opt/hydrogym/examples/jaxfluids/. . && chmod -R u+w .
python test_jaxfluids_env.py
'

# srun tests: one apptainer exec per rank, PMIx from Slurm (no --cleanenv: the
# ranks need the PMIX_*/SLURM_* variables; the image's own environment comes on top).
SRUN_EXEC=("${APPTAINER}" exec --nv --home "${RUN_DIR}/home" --bind "${RUN_DIR}"
    --env "HG_RUNS=${RUN_DIR}/runs,HF_HUB_OFFLINE=1,PYTHONUNBUFFERED=1,PYTHONPATH=,LD_PRELOAD=" "${SIF}")
srun_test() {
    case "$1" in
        srun_cpu)
            srun --mpi="${SRUN_MPI}" -n 4 -c 8 --cpu-bind=cores "${SRUN_EXEC[@]}" python -c "${ALLREDUCE_CPU}" ;;
        srun_gpu)
            srun --mpi="${SRUN_MPI}" -n 4 -c 8 --cpu-bind=cores "${SRUN_EXEC[@]}" python -c "${ALLREDUCE_GPU}" ;;
        maia_srun)
            local d="${RUN_DIR}/runs/maia_cyl_srun" pol=cylinder_2d_re200/ppo/seed1
            local ex=/opt/hydrogym/examples/maia/released_policies
            rm -rf "${d}"
            "${SRUN_EXEC[@]}" python "${ex}/prepare_run.py" --policy "${pol}" --work-dir "${d}" || return 1
            # MAIA_MPMD_APPNUM: the MPMD application number (srun --multi-prog sets no MPI_APPNUM);
            # %o: offset in the task range (0, 1) -> GPUs 0 and 1
            printf '0 %s\n1-2 %s\n' \
                "env MAIA_MPMD_APPNUM=0 ${SRUN_EXEC[*]} python ${ex}/run_policy.py --policy ${pol} --out run.npz --store_data --no-plot --actions 3 --start 3" \
                "env MAIA_MPMD_APPNUM=1 CUDA_VISIBLE_DEVICES=%o ${SRUN_EXEC[*]} maia properties_run.toml" > "${d}/mpmd.conf"
            (cd "${d}" && srun --mpi="${SRUN_MPI}" -n 3 -c 8 --cpu-bind=cores --multi-prog mpmd.conf) || return 1
            "${SRUN_EXEC[@]}" python -c "import numpy as np; d = np.load('${d}/run.npz'); print({k: np.round(d[k], 6).tolist() for k in d.files if k.endswith('C_D') or k.endswith('C_L')})"
            ;;
    esac
}

results=()
rc=0
for t in ${TESTS}; do
    if [[ -z "${CMD[$t]:-}" && " srun_cpu srun_gpu maia_srun " != *" $t "* ]]; then
        echo "unknown test '$t' (prepare ${ALL_TESTS})"; exit 1
    fi
    if [[ " srun_cpu srun_gpu maia_srun " == *" $t "* && -z "${SLURM_JOB_ID:-}" ]]; then
        results+=("$(printf '%-10s %-8s %6s  %s' "$t" SKIP - "needs a Slurm allocation")"); continue
    fi
    limit="${TIMEOUT:-900}"; [[ "$t" == nekrs ]] && limit="${TIMEOUT:-2400}"
    [[ "$t" == jaxfluids ]] && limit="${TIMEOUT:-240}"
    log="${LOG_DIR}/${t}.log"
    start=$(date +%s)
    if [[ " srun_cpu srun_gpu maia_srun " == *" $t "* ]]; then
        export -f srun_test; export SRUN_MPI RUN_DIR
        timeout -k 30 "${limit}" bash -c "$(declare -p SRUN_EXEC ALLREDUCE_CPU ALLREDUCE_GPU); srun_test $t" > "${log}" 2>&1
    else
        timeout -k 30 "${limit}" "${APPTAINER}" exec "${APPTAINER_ARGS[@]}" "${SIF}" bash -c "set -eo pipefail; ${CMD[$t]}" \
            > "${log}" 2>&1
    fi
    status=$?
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
    results+=("$(printf '%-10s %-8s %5ss  %s' "$t" "${res}" "${elapsed}" "${log}")")
    echo "${results[-1]}"
done

echo ""
echo "=== ${SIF}  ($(hostname)${SLURM_JOB_ID:+, job ${SLURM_JOB_ID}}, srun --mpi=${SRUN_MPI}) ==="
printf '%s\n' "${results[@]}"
exit "${rc}"
