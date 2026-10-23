#!/bin/bash
# Shared helper: ensure `hydrogym[<extras>]` is importable from a given venv.
#
# Every compute-backend feature (maia-gpu, maia-cpu, nek5000) calls this from
# its own postCreateCommand.sh after building its solver binary - otherwise
# the binary builds fine but there's no Python environment that can actually
# drive it via hydrogym, which defeats the point of the whole container.
#
# Each backend gets its OWN venv rather than sharing one: maia-gpu links
# against NVHPC/HPCX's MPI, maia-cpu/nek5000 against the base image's apt
# OpenMPI, and mpi4py is a compiled extension built against whichever mpicc
# is first on PATH at install time - mixing those in one venv risks an MPI
# ABI mismatch between hydrogym's mpi4py and the solver binary it's talking
# to over MPMD. firedrake/hydrogym-gpu keep their own separate venvs for the
# same reason (PETSc-linked mpi4py, and pip-managed CUDA, respectively).
#
# Usage: ensure_hydrogym.sh <venv_dir> <extras_csv>
# Idempotent: safe to call every postCreateCommand run.

set -euo pipefail

VENV_DIR="$1"
EXTRAS="$2"
HYDROGYM_DIR="/workspace"

if [[ ! -d "${HYDROGYM_DIR}" ]]; then
    echo "WARNING: ${HYDROGYM_DIR} not found - check the workspace bind mount in devcontainer.json."
    exit 0
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR}..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "${VENV_DIR}" --system-site-packages --seed
    else
        python3 -m venv --system-site-packages "${VENV_DIR}"
    fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Checked via importlib.metadata, not a bare `import hydrogym` - the
# workspaceFolder in every devcontainer config here is /workspace (the
# bind-mounted repo root itself), which contains a `hydrogym/` package
# subdirectory, and Python's CWD-based implicit namespace packages mean
# `import hydrogym` succeeds from that CWD purely because that directory
# exists, even with nothing installed - a false positive that made this
# skip a real `pip install`.
if python -c "import importlib.metadata; importlib.metadata.version('hydrogym')" 2>/dev/null; then
    echo "hydrogym already importable in ${VENV_DIR} - skipping install."
else
    echo "Installing hydrogym[${EXTRAS}] into ${VENV_DIR}..."
    (cd "${HYDROGYM_DIR}" && git lfs pull 2>/dev/null || true)

    # pip install here pulls ~1-2GB of wheels (vtk, tfp-nightly, etc.) in one
    # shot. Seen once in practice: pip died with "Illegal instruction (core
    # dumped)" partway through - not reproducible on retry, including a
    # --no-cache-dir --force-reinstall of the packages right where it died,
    # so it wasn't a real CPU/instruction-set incompatibility, just a
    # corrupted/truncated wheel (cached or downloaded). Retry a couple of
    # times, purging pip's wheel cache in between so a bad cached wheel
    # can't just get reused on the next attempt.
    MAX_ATTEMPTS=3
    attempt=1
    while true; do
        pip install -e "${HYDROGYM_DIR}[${EXTRAS}]" && break
        status=$?
        if (( attempt >= MAX_ATTEMPTS )); then
            echo "ERROR: pip install failed ${MAX_ATTEMPTS} times in a row (last exit code ${status}) - giving up." >&2
            exit "${status}"
        fi
        echo "WARNING: pip install failed (exit code ${status}, attempt ${attempt}/${MAX_ATTEMPTS})." \
             "Purging pip's wheel cache and retrying in 5s." >&2
        pip cache purge || true
        sleep 5
        attempt=$((attempt + 1))
    done
fi

python -c "import hydrogym, importlib.metadata; print(f'hydrogym {importlib.metadata.version(\"hydrogym\")} OK in ${VENV_DIR}')"
deactivate
