#!/usr/bin/env bash
#SBATCH --job-name=hg-wing
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --account=m-aia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5         # mpirun slots: deploy.py + NP = 4 NekRS ranks
#SBATCH --exclusive                 # the whole node: all 4 APUs and CPU cores
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Wing deployment of one controller on one MI300A node (4 APUs, NekRS NP=4) in the HydroGym ROCm image:
#   POLICY=zero RUN=runs/zero sbatch run_aac.sbatch
#   POLICY=zero RUN=runs/zero_trip30 TRIP_MODES=30 sbatch run_aac.sbatch
#   POLICY=td3:$HOME/policies/paper_s0/td3_final.zip RUN=runs/td3_s0 sbatch run_aac.sbatch
# Submit from this directory. The run stops with a chunk shortly before the time limit and the job resubmits
# itself until ACTIONS control actions are done (state.json). The Python package and the udf header come
# from this checkout (PYTHONPATH, NEKRS_UDF_INCLUDES), not from the copy in the image.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
HERE=$PWD
REPO=$(cd ../../.. && pwd)
SIF="${SIF:-$HOME/hydrogym-baselines/hydrogym-rocm.sif}"   # apptainer pull of docker.io/kl126162/hydrogym-rocm:rocm7.2.4-gfx942
POLICY="${POLICY:?set POLICY (zero | oc | ub | td3:<zip>)}"
RUN="${RUN:?set RUN (e.g. runs/zero)}"
ACTIONS="${ACTIONS:-2500}"
NP="${NP:-4}"
MAX_HOURS="${MAX_HOURS:-7.0}"     # deploy.py wall time (NekRS setup and the first JIT come on top)
TRIP_MODES="${TRIP_MODES:-}"     # spanwise modes of the trip (empty: the par's 60)
EXTRA_ARGS="${EXTRA_ARGS:-}"      # further deploy.py arguments, e.g. "--chunk 100 --alpha 0.5"

mkdir -p "$RUN"
actions_done() { python3 -c "import json; print(json.load(open('$RUN/state.json'))['actions'])" 2>/dev/null || echo 0; }
before=$(actions_done)
apptainer exec --rocm --cleanenv --bind "$REPO" --bind "$HOME" \
    --env PYTHONPATH="$REPO",NEKRS_BACKEND=HIP,OMP_NUM_THREADS=1${ROCR_VISIBLE_DEVICES:+,ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES} \
    "$SIF" bash -c "cd $HERE && NP=$NP RUN=$RUN TRIP_MODES=$TRIP_MODES ./run.sh --policy $POLICY --actions $ACTIONS --max-hours $MAX_HOURS $EXTRA_ARGS" \
    > "$RUN/deploy-${SLURM_JOB_ID:-local}.log" 2>&1
rc=$?
done=$(actions_done)
echo "exit $rc, $done of $ACTIONS actions ($before at the start of this job)"
# resubmit only if this job made progress (a failing setup must not loop)
if (( done < ACTIONS )) && (( done > before )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "resubmitting"
    POLICY="$POLICY" RUN="$RUN" ACTIONS="$ACTIONS" NP="$NP" MAX_HOURS="$MAX_HOURS" EXTRA_ARGS="$EXTRA_ARGS" TRIP_MODES="$TRIP_MODES" \
        SIF="$SIF" sbatch --export=ALL run_aac.sbatch
elif (( done < ACTIONS )); then
    echo "no progress in this job - not resubmitting, see $RUN/deploy-*.log"
fi
