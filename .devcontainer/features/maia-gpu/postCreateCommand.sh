#!/bin/bash
# MAIA GPU Feature Post-Create Command
#
# Runs after the container is created, once the host bind mount (see
# devcontainer.json "workspaceMount") has attached /workspace/wipmaiaml.
# This is where the actual configure/build happens (see install.sh for why
# it can't happen at image-build time). Idempotent: skips the build if a
# matching binary already exists.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/maia-gpu.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
PSTL_PRESET="${PSTL_PRESET:-ada}"
BUILD_TYPE="${BUILD_TYPE:-production}"
ENABLE_COMPONENTS="${ENABLE_COMPONENTS:-}"
DISABLE_COMPONENTS="${DISABLE_COMPONENTS:-}"

echo "=== MAIA GPU Post-Create Setup ==="

MAIA_DIR="/workspace/wipmaiaml"
BUILD_DIR="${MAIA_DIR}/build_nvhpc_${BUILD_TYPE}"

# wipmaiaml isn't public yet, so it only shows up here via the host bind
# mount. Once it's public, replace this guard with a `git clone` fallback
# (or clone it in install.sh at build time like the other features do).
if [[ ! -d "${MAIA_DIR}" ]]; then
    echo "WARNING: MAIA source not found at ${MAIA_DIR}."
    echo "Check that the host bind mount in devcontainer.json points at a valid wipmaiaml checkout."
    exit 0
fi

cd "${MAIA_DIR}"

# configure.py's host detection (cmake/GetHost.cmake) is purely based on the
# container's OS hostname pattern-matched against a hardcoded cluster list -
# it does NOT read MAIA_HOST/HOST env vars. This happens to still resolve to
# LocalGPU today since that's also the container's --hostname (set in
# devcontainer.json's runArgs), but that's coincidental, not load-bearing -
# pin it explicitly via the real override (see cmake/Configure.cmake) so it
# doesn't silently break if the hostname ever changes.
export MAIA_HOST_FILE="${MAIA_DIR}/auxiliary/hosts/LocalGPU.cmake"

if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
    echo "MAIA GPU binaries already present at ${BUILD_DIR} - skipping build."
else
    # A previous incomplete attempt can leave a stale CMakeCache.txt behind
    # pointing at old library paths - CMake reuses cached paths rather than
    # re-reading the host config on a plain re-run, so a partial build dir
    # must not be trusted.
    rm -rf "${BUILD_DIR}"

    echo "Initializing git submodules..."
    git submodule update --init --depth 1 -- include/Eigen include/cantera include/doctest include/hypre include/mfem include/sundials 2>/dev/null || true

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
    make -j16 -C "${BUILD_DIR}" maia test_maia

    if [[ -f "${BUILD_DIR}/bin/maia" && -f "${BUILD_DIR}/bin/test_maia" ]]; then
        echo "=== MAIA GPU build successful ==="
    else
        echo "ERROR: MAIA GPU build failed - binaries not found"
        exit 1
    fi

    echo "Running MAIA tests..."
    (cd "${BUILD_DIR}" && ./bin/test_maia)
fi

# Solver binary alone is useless without something that can drive it -
# make sure hydrogym[maia] is importable from a dedicated venv.
bash /workspace/hydrogym/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/maia-gpu maia

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
echo "  Build:         make -j16 -C ${BUILD_DIR} maia"
echo "  Test:          ${BUILD_DIR}/bin/test_maia"
echo "  Regression:    cd ${MAIA_DIR} && python3 auxiliary/dev_tools/regression_test.py run-and-compare --binary ${BUILD_DIR}/bin/maia --label my_test --against master_baseline --bc 2000 --steps 2000"
echo "  Profile:       cd ${MAIA_DIR} && python3 auxiliary/dev_tools/profile_run.py run --binary ${BUILD_DIR}/bin/maia --label my_profile --bc 2000"
echo "  Report:        cd ${MAIA_DIR} && python3 auxiliary/dev_tools/report.py --label my_test --baseline master_baseline"
echo ""
echo "=== MAIA GPU Post-Create Complete ==="
