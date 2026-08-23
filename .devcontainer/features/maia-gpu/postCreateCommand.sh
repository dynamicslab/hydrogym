#!/bin/bash
# MAIA GPU Feature Post-Create Command
#
# Runs after the container is created, once the host bind mount (see
# devcontainer.json "workspaceMount") has attached /workspace. This is where
# the actual configure/build happens (see install.sh for why it can't happen
# at image-build time). Idempotent: skips the build if a matching binary
# already exists.
#
# CAVEAT: builds against third_party/m-AIA, the private wipmaiaml dev tree
# (see install.sh's CAVEAT) - requires RWTH GitLab access to have been
# initialized on the host before this container was opened (see the
# submodule-init check below for what happens if it wasn't).

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/maia-gpu.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
PSTL_PRESET="${PSTL_PRESET:-ada}"
BUILD_TYPE="${BUILD_TYPE:-production}"
ENABLE_COMPONENTS="${ENABLE_COMPONENTS:-}"
DISABLE_COMPONENTS="${DISABLE_COMPONENTS:-}"
RUN_TESTS="${RUN_TESTS:-false}"

echo "=== MAIA GPU Post-Create Setup ==="

MAIA_DIR="/workspace/third_party/m-AIA"
BUILD_DIR="${MAIA_DIR}/build_nvhpc_${BUILD_TYPE}"

if [[ ! -d "${MAIA_DIR}" ]]; then
    echo "WARNING: MAIA source not found at ${MAIA_DIR}."
    echo "Check that the host bind mount in devcontainer.json points at a valid hydrogym checkout with submodules registered (.gitmodules)."
    exit 0
fi

# The submodule directory exists once .gitmodules is registered, but its
# content is only populated after `git submodule update --init` - an
# uninitialized submodule is just an empty dir, so check for that instead of
# re-cloning (this repo is already bind-mounted from the host; a fresh
# `git clone` here would be redundant with - and could conflict with -
# whatever's already checked out on the host side).
if [[ -z "$(ls -A "${MAIA_DIR}" 2>/dev/null)" ]]; then
    echo "Initializing third_party/m-AIA submodule..."
    # third_party/m-AIA is private (RWTH GitLab) - this container has no TTY
    # and no cached credentials of its own, so a `git submodule update --init`
    # that has to actually authenticate here will fail immediately rather
    # than prompt. Give a clear pointer instead of letting that raw auth
    # error be the only output.
    if ! git -C /workspace submodule update --init --depth 1 -- third_party/m-AIA; then
        echo "ERROR: could not initialize third_party/m-AIA (see git error above)."
        echo "This submodule is private and requires RWTH GitLab access. Run this on"
        echo "the HOST before opening the devcontainer, where it can prompt you for"
        echo "credentials interactively:"
        echo "  git submodule update --init --recursive -- third_party/m-AIA"
        exit 1
    fi
fi

echo "NOTE: third_party/m-AIA is the private wipmaiaml dev tree (RWTH GitLab)."

cd "${MAIA_DIR}"

# configure.py's host detection (cmake/GetHost.cmake) is purely based on the
# container's OS hostname pattern-matched against a hardcoded cluster list -
# it does NOT read MAIA_HOST/HOST env vars. MAIA_HOST_FILE is the actual
# override (see cmake/Configure.cmake) - it points straight at a host file
# and skips hostname detection entirely.
DEVCONTAINER_HOST_FILE="${MAIA_DIR}/auxiliary/hosts/DEVCONTAINER.cmake"

# third_party/m-AIA doesn't ship this file - it's genuinely
# environment-specific (NVHPC/CUDA/library paths baked in by the base
# image), the same category of thing as any other machine-specific MAIA
# host config. Materialize it from the template in this repo if it isn't
# already present in the submodule (if wipmaiaml ever starts shipping one,
# this becomes a no-op and the template copy below is skipped entirely).
if [[ ! -f "${DEVCONTAINER_HOST_FILE}" ]]; then
    echo "Materializing DEVCONTAINER.cmake host config from template..."
    mkdir -p "$(dirname "${DEVCONTAINER_HOST_FILE}")"
    cp /workspace/.devcontainer/base/DEVCONTAINER.cmake "${DEVCONTAINER_HOST_FILE}"
fi

# The template's PSTL value is just its own default - always sync it to
# whatever architecture was actually requested via the pstlPreset feature
# option, regardless of where the host file came from.
sed -i "s/set(PSTL \"[^\"]*\")/set(PSTL \"${PSTL_PRESET}\")/" "${DEVCONTAINER_HOST_FILE}"

export MAIA_HOST_FILE="${DEVCONTAINER_HOST_FILE}"

if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
    echo "MAIA GPU binaries already present at ${BUILD_DIR} - skipping build."
else
    # A previous incomplete attempt can leave a stale CMakeCache.txt behind
    # pointing at old library paths - CMake reuses cached paths rather than
    # re-reading the host config on a plain re-run, so a partial build dir
    # must not be trusted.
    rm -rf "${BUILD_DIR}"

    echo "Initializing MAIA's own git submodules (Eigen, cantera, doctest, hypre, mfem, sundials)..."
    # These are all public repos (unlike include/PyJacReactionMechanisms,
    # deliberately excluded here - private RWTH repo, only needed if
    # ENABLE_PYJAC is on) - a failure here is most likely transient
    # (network), not an auth problem. Fail loudly instead of silently
    # swallowing it and letting configure.py fail later with a much more
    # confusing "Eigen not found"-style error.
    if ! git submodule update --init --depth 1 -- include/Eigen include/cantera include/doctest include/hypre include/mfem include/sundials; then
        echo "ERROR: could not initialize MAIA's own submodules (see git error above)."
        echo "Retry, or run this manually from ${MAIA_DIR}:"
        echo "  git submodule update --init --depth 1 -- include/Eigen include/cantera include/doctest include/hypre include/mfem include/sundials"
        exit 1
    fi

    CONFIGURE_CMD=(
        python3 configure.py nvhpc "${BUILD_TYPE}"
        --enable-pstl "${PSTL_PRESET}"
        --with-hdf5
        --disable-updateGitSubmodules
        --compile-commands
    )

    if [[ -n "${ENABLE_COMPONENTS}" ]]; then
        IFS=',' read -ra COMPS <<< "${ENABLE_COMPONENTS}"
        for comp in "${COMPS[@]}"; do
            CONFIGURE_CMD+=(--enable-"${comp}")
        done
    fi

    if [[ -n "${DISABLE_COMPONENTS}" ]]; then
        IFS=',' read -ra COMPS <<< "${DISABLE_COMPONENTS}"
        for comp in "${COMPS[@]}"; do
            CONFIGURE_CMD+=(--disable-"${comp}")
        done
    fi

    echo "Running configure.py..."
    "${CONFIGURE_CMD[@]}"

    echo "Building MAIA (solver + tests)..."
    MAIA_BUILD_JOBS="$(compute_parallelism.sh)"
    echo "Using -j${MAIA_BUILD_JOBS} (RAM-aware, see compute_parallelism.sh)"
    make -j"${MAIA_BUILD_JOBS}" -C "${BUILD_DIR}" maia test_maia

    if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
        echo "=== MAIA GPU build successful ==="
    else
        echo "ERROR: MAIA GPU build failed - binaries not found"
        exit 1
    fi

    if [[ "${RUN_TESTS}" == "true" ]]; then
        echo "Running MAIA tests..."
        # Non-fatal: test_lb_latticedescriptor.cpp:77 (oppositeDist inside a
        # GPU-offloaded parallelFor) is a known pre-existing crash, confirmed
        # on plain master too (not a wipmaiaml issue) - it's
        # architecture/driver sensitive (cudaErrorIllegalAddress on one GPU,
        # cudaErrorInvalidDevice on another) and unresolved. Letting it kill
        # this script under set -e would abort hydrogym venv setup etc.
        # below even though the solver binary above already built
        # successfully and is fully usable.
        test_exit=0
        (cd "${BUILD_DIR}" && ./bin/test_maia) || test_exit=$?
        if [[ ${test_exit} -ne 0 ]]; then
            echo "WARNING: test_maia exited ${test_exit} - continuing setup anyway (solver build above succeeded)."
            echo "If the only failure is test_lb_latticedescriptor.cpp:77, that's the known pre-existing GPU parallelFor crash - not a build problem."
        fi
    else
        echo "Skipping MAIA tests (runTests feature option is off by default - set it to true to enable)."
    fi
fi

# Solver binary alone is useless without something that can drive it -
# make sure hydrogym[maia] is importable from a dedicated venv.
bash /workspace/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/maia-gpu maia

# Set up VS Code compile_commands.json symlink if needed
if [[ -f "${BUILD_DIR}/compile_commands.json" && ! -L "${MAIA_DIR}/compile_commands.json" ]]; then
    ln -sf "${BUILD_DIR}/compile_commands.json" "${MAIA_DIR}/compile_commands.json"
    echo "Created compile_commands.json symlink"
fi

# Verify dev_tools are available
for tool in regression_test.py profile_run.py report.py; do
    if [[ -f "${MAIA_DIR}/auxiliary/dev_tools/${tool}" ]]; then
        echo "${tool} verified"
    fi
done

# Verify dev_runs directory structure
if [[ -d "/workspace/dev_runs" ]]; then
    echo "dev_runs directory verified"
else
    echo "Creating dev_runs directory structure..."
    mkdir -p /workspace/dev_runs/scratch/regression
    mkdir -p /workspace/dev_runs/scratch/profile
    mkdir -p /workspace/dev_runs/results/regression
    mkdir -p /workspace/dev_runs/results/profiling
fi

echo ""
echo "=== Development Environment Ready ==="
echo "MAIA Binary:     ${BUILD_DIR}/bin/maia"
echo "Test Binary:     ${BUILD_DIR}/bin/test_maia"
echo "Dev Tools:       ${MAIA_DIR}/auxiliary/dev_tools/"
echo "Dev Runs:        /workspace/dev_runs/"
echo ""
echo "Quick commands:"
echo "  Build:         make -j\"\$(compute_parallelism.sh)\" -C ${BUILD_DIR} maia"
echo "  Test:          ${BUILD_DIR}/bin/test_maia"
echo "  Regression:    cd ${MAIA_DIR} && python3 auxiliary/dev_tools/regression_test.py run-and-compare --binary ${BUILD_DIR}/bin/maia --label my_test --against master_baseline --bc 2000 --steps 2000"
echo "  Profile:       cd ${MAIA_DIR} && python3 auxiliary/dev_tools/profile_run.py run --binary ${BUILD_DIR}/bin/maia --label my_profile --bc 2000"
echo "  Report:        cd ${MAIA_DIR} && python3 auxiliary/dev_tools/report.py --label my_test --baseline master_baseline"
echo ""
echo "=== MAIA GPU Post-Create Complete ==="
