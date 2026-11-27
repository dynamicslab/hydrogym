#!/usr/bin/env bash
# Prepare, run and plot one released policy against the uncontrolled flow (m-AIA, MPMD).
#
# Usage:
#   ./run_example.sh <policy-id> <work-dir> [n-maia-ranks (default 1)] [run_policy.py options ...]
#
# run_policy.py options: --actions N, --start K, --runs unc|ctl, --file-role ROLE (also given to prepare_run.py),
# --readable (also the readable figures and the actions), --store_data (also run.npz and policy_run.json). The
# figures (run.png; with --readable also run_readable.png, run_actions_readable.png) are made by run_policy.py.
#
# Examples:
#   ./run_example.sh cylinder_2d_re200/ppo/seed1 runs/cyl200 1
#   ./run_example.sh pinball_2d_re100/ppo_tuned/seed0 runs/pin100 1 --start 6 --readable --store_data
#   ./run_example.sh naca0012_gust_re1000/td3_3jet_jet1.5U/seed1 runs/gust3d 4
#
# Environment variables:
#   MAIA_ARGS     arguments of m-AIA after the property file (default "--silent": no per-step solver output)
#   MPIRUN        MPMD launcher (default "mpirun"; e.g. "mpirun --bind-to none" or "prterun --bind-to none")
#   MAIA          m-AIA executable (default "maia")
#   PYTHON        Python interpreter (default "python")
#   PREPARE_ARGS  extra options of prepare_run.py (e.g. "--dim 3", "--env-dir /path/to/<Environment>")
#   PLOT          0 = no figures, store run.npz instead (e.g. on a cluster without matplotlib; plot it elsewhere)
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,21p' "$0"
    exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="$1"
WORK="$2"
NP="${3:-1}"  # m-AIA ranks: 1 per GPU in the GPU image (2D); 3D: one rank per GPU, e.g. 4
shift $(( $# < 3 ? $# : 3 ))
PY="${PYTHON:-python}"
ROLE=()  # --file-role is also needed by prepare_run.py (settings taken from the policy file)
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--file-role" ]; then ROLE=(--file-role "${ARGS[$((i + 1))]}"); fi
done

# shellcheck disable=SC2086
"$PY" "$HERE/prepare_run.py" --policy "$POLICY" --work-dir "$WORK" "${ROLE[@]}" ${PREPARE_ARGS:-}
PLOT_ARGS=()
if [ "${PLOT:-1}" = "0" ]; then PLOT_ARGS=(--no-plot --store_data); fi
cd "$WORK"
# shellcheck disable=SC2086
${MPIRUN:-mpirun} -np 1 "$PY" "$HERE/run_policy.py" --policy "$POLICY" --out run.npz "$@" "${PLOT_ARGS[@]}" \
    : -np "$NP" "${MAIA:-maia}" properties_run.toml ${MAIA_ARGS:---silent}
