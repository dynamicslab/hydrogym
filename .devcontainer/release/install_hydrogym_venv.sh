#!/bin/bash
# The HydroGym venvs of the release images, in two steps so that a new
# HydroGym revision only re-runs the second, small one:
#
#   MODE=deps     (needs only pyproject.toml + uv.lock in HYDROGYM_SRC)
#                 /opt/venvs/hydrogym: every dependency of
#                 hydrogym[maia,nek,nekrs,jax,jaxfluids] at the version of the
#                 release's uv.lock, torch (CPU build: its CUDA wheels would add
#                 several GB and a second CUDA stack), JAX (JAX_SPEC, at the
#                 uv.lock version), JAX-Fluids, and mpi4py built from source
#                 against the image's MPI, so the Python controller and the
#                 solver ranks of an MPMD launch load the same libmpi;
#                 /opt/venvs/firedrake: the hydrogym[firedrake] dependencies,
#                 Stable-Baselines3, TensorBoard and torch (CPU); the packages
#                 it shares with the lock at their uv.lock versions.
#   MODE=project  HydroGym itself (editable, from HYDROGYM_SRC) into both venvs,
#                 without dependencies.
#
# Environment:
#   HYDROGYM_SRC   HydroGym source
#   VENV           /opt/venvs/hydrogym
#   MPI_CC         MPI C compiler wrapper for mpi4py (deps)
#   JAX_SPEC       e.g. "jax[cuda12]" or "jax" (CPU)
#   TORCH_INDEX    https://download.pytorch.org/whl/cpu
#   JAXFLUIDS_REF  JAX-Fluids git commit
#   FIREDRAKE_VENV /opt/venvs/firedrake (skipped if missing)

set -euo pipefail

: "${HYDROGYM_SRC:?}" "${MODE:?}"
VENV="${VENV:-/opt/venvs/hydrogym}"
JAX_SPEC="${JAX_SPEC:-jax}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
JAXFLUIDS_REF="${JAXFLUIDS_REF:-819edcd2f496e1719d1f7db751d06b9fd9a1f3cc}"
FIREDRAKE_VENV="${FIREDRAKE_VENV:-/opt/venvs/firedrake}"
PYTHON="${PYTHON:-/usr/bin/python3}"
EXTRAS="maia,nek,nekrs,jax,jaxfluids"

lock_export() {
    (cd "${HYDROGYM_SRC}" && uv export --frozen --no-hashes --no-emit-project --no-dev "$@")
}

if [[ "${MODE}" == "deps" ]]; then
    : "${MPI_CC:?}"
    work="$(mktemp -d)"
    # shellcheck disable=SC2046
    lock_export $(printf -- '--extra %s ' ${EXTRAS//,/ }) > "${work}/all.txt"
    grep -v -E '^(torch|triton|nvidia-)' "${work}/all.txt" > "${work}/requirements.txt"
    torch_version="$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' "${work}/all.txt" | head -1)"
    jax_version="$(sed -n "s/^jax==\([^ ;]*\) ; python_full_version >= '3.12'.*/\1/p" "${work}/all.txt" | head -1)"
    [[ -n "${jax_version}" ]] || jax_version="$(sed -n 's/^jax==\([^ ;]*\)$/\1/p' "${work}/all.txt" | head -1)"
    : "${torch_version:?torch not in uv.lock}" "${jax_version:?jax not in uv.lock}"
    echo "torch ${torch_version} (CPU), ${JAX_SPEC}==${jax_version}"

    uv venv "${VENV}" --python "${PYTHON}" --seed
    VIRTUAL_ENV="${VENV}" uv pip install --no-cache torch=="${torch_version}" --index-url "${TORCH_INDEX}"
    # Pinned requirements, then JAX's backend at the same version and JAX-Fluids
    # (which needs nothing beyond them).
    VIRTUAL_ENV="${VENV}" MPICC="${MPI_CC}" uv pip install --no-cache --no-binary mpi4py -r "${work}/requirements.txt"
    VIRTUAL_ENV="${VENV}" uv pip install --no-cache -c "${work}/requirements.txt" "${JAX_SPEC}==${jax_version}" \
        "jaxfluids @ git+https://github.com/tumaer/JAXFLUIDS.git@${JAXFLUIDS_REF}"
    ln -sfn "$(basename "${VENV}")" "$(dirname "${VENV}")/nekrs"
    "${VENV}/bin/python" -c "import mpi4py, jax, jaxfluids, torch; \
print('jax', jax.__version__, 'torch', torch.__version__, '- mpi4py built with', mpi4py.get_config())"

    if [[ -x "${FIREDRAKE_VENV}/bin/python" ]]; then
        # hydrogym and hydrogym[firedrake] dependencies, resolved against what
        # the Firedrake build installed (numpy, mpi4py, h5py, petsc4py stay).
        # vtk: Firedrake's VTK output, used by the HydroGym Firedrake flows.
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip install --no-cache -r "${HYDROGYM_SRC}/pyproject.toml" --extra firedrake vtk
        # Stable-Baselines3 with TensorBoard and torch (CPU build) for the
        # Firedrake RL examples, at the versions of /opt/venvs/hydrogym; the
        # packages they add come at their uv.lock versions too (constraints:
        # uv.lock without what the venv already has).
        norm() { tr 'A-Z_.' 'a-z--'; }
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip freeze | sed -E 's/[ =@;].*//' | norm | sort -u > "${work}/fd_installed.txt"
        awk -F'==' 'NR == FNR { have[$1]; next } /==/ { n = tolower($1); gsub(/[_.]/, "-", n); if (!(n in have)) print }' \
            "${work}/fd_installed.txt" "${work}/requirements.txt" > "${work}/fd_constraints.txt"
        sb3_version="$(sed -n 's/^stable-baselines3==\([^ ;]*\).*/\1/p' "${work}/all.txt" | head -1)"
        tb_version="$(sed -n 's/^tensorboard==\([^ ;]*\).*/\1/p' "${work}/all.txt" | head -1)"
        : "${sb3_version:?stable-baselines3 not in uv.lock}" "${tb_version:?tensorboard not in uv.lock}"
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip install --no-cache torch=="${torch_version}" --index-url "${TORCH_INDEX}"
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip install --no-cache -c "${work}/fd_constraints.txt" \
            stable-baselines3=="${sb3_version}" tensorboard=="${tb_version}"
        # Every package shared with the lock at its uv.lock version (the lock
        # entries whose markers hold for this interpreter), except what
        # build_firedrake.sh compiled (petsc4py, mpi4py, h5py) and decorator
        # (Firedrake requires decorator<=4.4.2).
        lock_export --all-extras > "${work}/fd_export.txt"
        "${FIREDRAKE_VENV}/bin/python" - "${work}/fd_export.txt" > "${work}/fd_lock.txt" <<'PY'
import sys
from packaging.markers import Marker
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith(("#", "-")) or "==" not in line:
        continue
    req, _, marker = line.partition(";")
    if marker.strip() and not Marker(marker.strip()).evaluate():
        continue
    print(req.strip())
PY
        keep='^(petsc4py|mpi4py|h5py|decorator|firedrake|hydrogym|torch)$'
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip freeze | grep '==' | sed 's/ .*//' > "${work}/fd_now.txt"
        awk -F'==' -v keep="${keep}" 'NR == FNR { n = tolower($1); gsub(/[_.]/, "-", n); lock[n] = $2; next }
            { n = tolower($1); gsub(/[_.]/, "-", n); if ((n in lock) && lock[n] != $2 && n !~ keep) print n "==" lock[n] }' \
            "${work}/fd_lock.txt" "${work}/fd_now.txt" > "${work}/fd_align.txt"
        grep -v -i -E '^(petsc4py|mpi4py|h5py|decorator|firedrake|hydrogym|torch)==' "${work}/fd_lock.txt" > "${work}/fd_lock_cons.txt"
        echo "firedrake venv: $(wc -l < "${work}/fd_align.txt") packages to their uv.lock versions"
        if [[ -s "${work}/fd_align.txt" ]]; then
            VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip install --no-cache -c "${work}/fd_lock_cons.txt" -r "${work}/fd_align.txt"
        fi
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip check
        OMP_NUM_THREADS=1 "${FIREDRAKE_VENV}/bin/python" -c "import firedrake, stable_baselines3, tensorboard, torch; \
print('firedrake venv: stable-baselines3', stable_baselines3.__version__, 'torch', torch.__version__)"
    fi
    rm -rf "${work}"
elif [[ "${MODE}" == "project" ]]; then
    VIRTUAL_ENV="${VENV}" uv pip install --no-cache --no-deps -e "${HYDROGYM_SRC}"
    "${VENV}/bin/python" -c "import hydrogym, importlib.metadata as m; print('hydrogym', m.version('hydrogym'), 'in ${VENV}')"
    if [[ -x "${FIREDRAKE_VENV}/bin/python" ]]; then
        VIRTUAL_ENV="${FIREDRAKE_VENV}" uv pip install --no-cache --no-deps -e "${HYDROGYM_SRC}"
    fi
else
    echo "ERROR: MODE must be deps or project, got '${MODE}'"; exit 1
fi
