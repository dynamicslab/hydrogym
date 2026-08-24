#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
firedrake_python="${FIREDRAKE_PYTHON:-/home/firedrake/firedrake/bin/python}"

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
uv pip install --python "${firedrake_python}" --require-hashes --requirement "${locked_requirements}"
uv pip install --python "${firedrake_python}" --no-deps --editable "${project_root}"
