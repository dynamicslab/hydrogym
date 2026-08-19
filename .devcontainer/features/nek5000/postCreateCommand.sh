#!/bin/bash
# Nek5000 Feature Post-Create Command
#
# Builds each requested case's nek5000 binary from the in-repo
# third_party/nek5000/ tree:
#   third_party/nek5000/solver/KTH_DRL_Framework/Nek5000   (submodule, core solver)
#   third_party/nek5000/solver/KTH_DRL_Framework/Toolbox   (submodule, KTH DRL toolbox)
#   third_party/nek5000/solver/DEC_DRL_Nek5000/             (vendored copy, different,
#                                                             older Nek5000 core - see
#                                                             its VENDORED_README.md)
#   third_party/nek5000/cases/<case>/                       (case source, committed)
#
# Two build flavors, mirroring the case's own committed compile_script. The
# two cores above are NOT interchangeable (see DEC_DRL_Nek5000/VENDORED_README.md
# for the structural reasons why) - each flavor must stay pointed at its own:
#   - "kth":  mini_channel, large_channel - links against the KTH Toolbox
#             for the frame/mntrlog/rprm/tsrs DRL hooks (nekconfig -build-dep
#             + makenek + make, matching the recipe that used to live in the
#             Nek5000-1.0-gompi-*.eb EasyBuild module - makenek alone is known
#             to be flaky here, hence the explicit `make` afterwards). Uses
#             NEK_CORE (KTH_DRL_Framework/Nek5000).
#   - "dec":  naca0012_200k, small_wing - self-contained case source (no
#             Toolbox dependency), built via the case's own compile_script,
#             pointed at the vendored DEC_DRL_Nek5000 core via its override
#             arg. Uses NEK_CORE_DEC (DEC_DRL_Nek5000) - a distinctly-named
#             variable from NEK_CORE on purpose, so the two can never be
#             silently swapped in a future edit.
#
# Runs after the container is created, once the host bind mount (see
# devcontainer.json "workspaceMount") has attached /workspace.
# Idempotent per case: skips a case if its binary already exists.

set -euo pipefail

CONFIG_FILE=/opt/maia-feature-config/nek5000.env
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi
CASES="${CASES:-mini_channel,large_channel,naca0012_200k,small_wing}"

echo "=== Nek5000 Post-Create Setup ==="

NEK_ROOT="/workspace/third_party/nek5000"
NEK_CORE="${NEK_ROOT}/solver/KTH_DRL_Framework/Nek5000"
TOOLBOX_SRC="${NEK_ROOT}/solver/KTH_DRL_Framework/Toolbox"
NEK_CORE_DEC="${NEK_ROOT}/solver/DEC_DRL_Nek5000"
CASES_DIR="${NEK_ROOT}/cases"

if [[ ! -d "${NEK_CORE}/bin" ]]; then
    echo "WARNING: Nek5000 core not found at ${NEK_CORE}."
    echo "Run 'git submodule update --init third_party/nek5000/solver/KTH_DRL_Framework/Nek5000 third_party/nek5000/solver/KTH_DRL_Framework/Toolbox' in the hydrogym repo on the host, then rebuild/reopen the container."
    exit 0
fi

if [[ ! -d "${NEK_CORE_DEC}/core" ]]; then
    echo "WARNING: DEC_DRL_Nek5000 core not found at ${NEK_CORE_DEC}."
    echo "This is a vendored copy (not a submodule) - it should already be committed at third_party/nek5000/solver/DEC_DRL_Nek5000/. Check the checkout."
    exit 0
fi

chmod -R +x "${NEK_CORE}/bin" "${NEK_CORE}/core" "${TOOLBOX_SRC}" "${NEK_CORE_DEC}/core" 2>/dev/null || true

build_kth_case() {
    local case_dir="$1" case_name="$2" pplist="$3" usr="$4"

    cd "${case_dir}"
    export NEK_SOURCE_ROOT="${NEK_CORE}"
    export TOOLBOX_SRC="${TOOLBOX_SRC}"
    # shellcheck disable=SC1091
    source "${TOOLBOX_SRC}/toolbox_path.sh"
    # The vendored case makefile_usr.inc files reference $(MONITOR_SRC) and
    # $(RUNPARAM_SRC), but this Toolbox fork's toolbox_path.sh exports the
    # same directories under different names (MNTR_SRC, RPRM_SRC) - alias
    # them rather than editing the vendored case files.
    export MONITOR_SRC="${MNTR_SRC}"
    export RUNPARAM_SRC="${RPRM_SRC}"
    export CASE="${case_name}"
    # Explicit paths, not bare mpif77/mpicc - those resolve to NVHPC's
    # compilers since HPCX is first on PATH (see base/Dockerfile.base).
    # makefile_usr.inc's own rules for mntrlog.o/mntrtmr.o/rprm.o etc invoke
    # $(F77) directly (not $(FC)) with no other -I flags of their own, so
    # F77 needs the same TOOLBOX_INC include paths baked in as FC - without
    # them, compiling Toolbox sources fails with "Cannot open included file
    # 'FRAMELP'" (it lives under Toolbox/driver/frame, one of TOOLBOX_INC's
    # -I entries).
    export FC="/usr/bin/mpifort.openmpi ${TOOLBOX_INC:-}"
    export F77="/usr/bin/mpifort.openmpi ${TOOLBOX_INC:-}"
    export CC=/usr/bin/mpicc.openmpi
    # Belt-and-suspenders: makenek's generated Makefile/makefile_usr.inc can
    # reference $(F77) resolved from a bare `mpif77` at generation time
    # rather than the FC/F77 exported above (same PATH-order trap as
    # elsewhere - HPCX's nvfortran is first on PATH, and it doesn't
    # understand gfortran flags like -fdefault-real-8/-std=legacy).
    export PATH="/usr/bin:${PATH}"
    export PPLIST="${pplist}"
    export USR="${usr}"
    export USR_LFLAGS=""
    unset CFLAGS FFLAGS CXXFLAGS LDFLAGS CPPFLAGS

    # makenek.inc's "clean"/"-build-dep" paths prompt interactively
    # ("do you want to clean/rebuild all 3rd party dependencies too? [N]")
    # with no way to pass a default non-interactively - </dev/null makes
    # `read` hit EOF immediately instead of hanging forever waiting for a
    # TTY that doesn't exist in this context; empty input falls through to
    # the shown [N] default either way.
    "${NEK_SOURCE_ROOT}/bin/nekconfig" -build-dep < /dev/null
    "${NEK_SOURCE_ROOT}/bin/makenek" clean < /dev/null 2>&1 | tail -20 || true
    "${NEK_SOURCE_ROOT}/bin/makenek" "${CASE}" < /dev/null 2>&1 | tail -20 || true
    make -j4 2>&1 | tee make.log
}

build_dec_case() {
    local case_dir="$1" case_name="$2"

    cd "${case_dir}"
    chmod +x compile_script
    # compile_script hardcodes bare CC="mpicc"/F77="mpif77" internally (it's
    # vendored case source, left untouched) - putting /usr/bin first on PATH
    # makes those resolve to the apt/GNU OpenMPI wrapper (via
    # update-alternatives) instead of NVHPC's, which is first on PATH by
    # default (see base/Dockerfile.base).
    # $2 overrides compile_script's default SOURCE_ROOT
    # (../../solver/DEC_DRL_Nek5000, which already resolves to the same
    # place) - passed explicitly so this never silently drifts if either
    # path changes. Unlike the KTH path, DEC_DRL_Nek5000's makenek.inc runs
    # unconditionally (no config_nek, no interactive 3rd-party-deps prompt),
    # so no </dev/null guard is needed here.
    PATH="/usr/bin:${PATH}" bash compile_script "${case_name}" "${NEK_CORE_DEC}"
}

IFS=',' read -ra CASE_LIST <<< "${CASES}"
for case in "${CASE_LIST[@]}"; do
    CASE_DIR="${CASES_DIR}/${case}"

    if [[ ! -d "${CASE_DIR}" ]]; then
        echo "WARNING: unknown case '${case}' (no directory at ${CASE_DIR}), skipping."
        continue
    fi

    if [[ -f "${CASE_DIR}/nek5000" ]]; then
        echo "${case}: binary already present at ${CASE_DIR}/nek5000 - skipping build."
        continue
    fi

    echo "=== Building Nek5000 case: ${case} ==="
    # Each case builds in its own subshell: build_kth_case exports FC/CC/
    # TOOLBOX_INC etc (needed since Nek5000's generated makefile reads them
    # via $(FC) for core files, not just via the compile_script's own local
    # vars), and without a subshell those exports leak into the next case's
    # build via this script's own environment - e.g. a DEC-flavor case
    # picking up the previous KTH case's Toolbox -I flags and double-
    # declaring symbols already provided by its own core includes.
    case "${case}" in
        mini_channel)
            ( build_kth_case "${CASE_DIR}" phill "MPIIO DRL TSRS" \
                "frame.o mntrlog_block.o mntrlog.o mntrtmr_block.o mntrtmr.o rprm_block.o rprm.o io_tools_block.o io_tools.o chkpoint.o chkpt_mstp.o comm_mpi_tool.o map2D.o stat.o stat_IO.o math_tools.o misc.o drl_main.o drl_state.o drl_reward.o drl_IO.o drl_action.o drl_init.o tsrs.o tsrs_IO.o pts_redistribute.o" )
            ;;
        large_channel)
            ( build_kth_case "${CASE_DIR}" tcf "MPIIO DRL BDFD TSRS" \
                "frame.o mntrlog_block.o mntrlog.o mntrtmr_block.o mntrtmr.o rprm_block.o rprm.o io_tools_block.o io_tools.o chkpoint.o chkpt_mstp.o comm_mpi_tool.o map2D.o stat.o stat_IO.o math_tools.o drl_main.o drl_state.o drl_reward.o drl_IO.o drl_action.o drl_init.o tsrs.o tsrs_IO.o pts_redistribute.o bdforce.o" )
            ;;
        naca0012_200k)
            ( build_dec_case "${CASE_DIR}" naca_wing )
            ;;
        small_wing)
            ( build_dec_case "${CASE_DIR}" small_wing )
            ;;
        *)
            echo "WARNING: no build recipe registered for case '${case}', skipping."
            continue
            ;;
    esac

    if [[ -f "${CASE_DIR}/nek5000" ]]; then
        echo "=== ${case} build successful: ${CASE_DIR}/nek5000 ==="
    else
        echo "ERROR: ${case} build failed - nek5000 binary not found"
        exit 1
    fi
done

# Solver binaries alone are useless without something that can drive them -
# make sure hydrogym[nek] is importable from a dedicated venv.
bash /workspace/.devcontainer/scripts/ensure_hydrogym.sh /opt/venvs/nek5000 nek

echo ""
echo "=== Nek5000 Post-Create Complete ==="
echo "Case binaries: ${CASES_DIR}/<case>/nek5000"
echo "Add a case's directory to PATH, or pass --nek-path explicitly, to use it from hydrogym.nek."
