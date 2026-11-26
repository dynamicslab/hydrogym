#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The Firedrake interpreter: FIREDRAKE_PYTHON, else the active virtual environment (e.g. after
# `source /opt/venvs/firedrake/bin/activate` in the HydroGym CPU image), else the old Firedrake image path.
if [[ -n "${FIREDRAKE_PYTHON:-}" ]]; then
    firedrake_python="${FIREDRAKE_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    firedrake_python="${VIRTUAL_ENV}/bin/python"
else
    firedrake_python="/home/firedrake/firedrake/bin/python"
fi

if [[ ! -x "${firedrake_python}" ]]; then
    echo "Firedrake Python is not executable: ${firedrake_python}" >&2
    echo "Set FIREDRAKE_PYTHON to the existing Firedrake environment interpreter." >&2
    exit 1
fi

locked_requirements="$(mktemp)"
trap 'rm -f "${locked_requirements}"' EXIT

export_args=(
    --project "${project_root}"
    --locked
    --no-default-groups
    --extra firedrake
    --no-emit-project
    --output-file "${locked_requirements}"
)

if [[ "${1:-}" == "--dev" ]]; then
    export_args+=(--group dev)
fi

uv export "${export_args[@]}"
# --no-config: uv pip would otherwise apply [tool.uv] of the project in the working directory (e.g. the unpinned
# soupsieve constraint-dependencies), which --require-hashes rejects
uv pip install --no-config --python "${firedrake_python}" --require-hashes --requirement "${locked_requirements}"
uv pip install --no-config --python "${firedrake_python}" --no-deps --editable "${project_root}"
