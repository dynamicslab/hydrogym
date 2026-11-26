#!/usr/bin/env bash
#SBATCH --job-name=hg-blstats
# add "#SBATCH --partition=<partition>" and "#SBATCH --account=<project>" here if the cluster requires them
#SBATCH --nodes=1
#SBATCH --exclusive                 # all CPU cores and the node memory (about 2.5 GB per worker and run)
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Boundary-layer statistics of the uncontrolled wing runs on the CPUs of one node, in a HydroGym image (see ../slurm_image.sh for SIF):
# bl_stats.py per case (in parallel, RUN/field_stats.npz from the restart fields after SKIP actions), then
# bl_report.py over all cases (OUT/report.md, summary.json, CSV per run, figures):
#   sbatch bl_stats_slurm.sbatch
#   CASES="runs/zero_trip60:500 runs/zero_ref1:2420" WORKERS=16 OUT=bl_report sbatch bl_stats_slurm.sbatch
#   REPORT_ONLY=1 sbatch bl_stats_slurm.sbatch     # field_stats.npz exist: only the report
# Submit from this directory.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
HERE=$PWD
REPO=$(cd ../../.. && pwd)
# shellcheck source=../slurm_image.sh
source ../slurm_image.sh       # SIF (CPU only here: the image of the runs, no GPU flag needed)
CASES="${CASES:-runs/zero_trip60:500 runs/zero_ref1:2420}"   # RUN:SKIP (actions left out as transient)
WORKERS="${WORKERS:-16}"          # fields processed at once per case
OUT="${OUT:-bl_report}"
REPORT_ONLY="${REPORT_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"      # further bl_stats.py arguments, e.g. "--stations 0.02 0.98 0.005"

export APPTAINERENV_PYTHONPATH="$REPO" APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
BIND="${BIND:-}"                 # further directory to bind into the image
PY="apptainer exec --cleanenv --bind $REPO ${BIND:+--bind $BIND} $SIF /opt/venvs/nekrs/bin/python"
$PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" || exit 1
$PY -c "import matplotlib" 2>/dev/null || echo "no matplotlib in the image: report without figures"

runs=() skips=()
for c in $CASES; do
    runs+=("${c%%:*}") skips+=("${c##*:}")
done
if (( REPORT_ONLY == 0 )); then
    pids=()
    for i in "${!runs[@]}"; do
        $PY bl_stats.py "${runs[$i]}" --skip "${skips[$i]}" --workers "$WORKERS" $EXTRA_ARGS \
            > "${runs[$i]}/bl_stats-${SLURM_JOB_ID:-local}.log" 2>&1 &
        pids+=($!)
    done
    rc=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "bl_stats.py ${runs[$i]} failed, see ${runs[$i]}/bl_stats-*.log"; rc=1; }
    done
    (( rc == 0 )) || echo "continuing with the report of the wall samples (field statistics missing for the failed cases)"
fi
$PY bl_report.py "${runs[@]}" --skip "${skips[@]}" --out "$OUT"
echo "done: $HERE/$OUT/report.md"
