#!/usr/bin/env bash
#SBATCH --job-name=hg-wing
# add "#SBATCH --partition=<partition>" and "#SBATCH --account=<project>" here if the cluster requires them
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9         # deploy.py + up to NP = 8 NekRS ranks
#SBATCH --exclusive                 # the whole node: all GPUs and CPU cores
#SBATCH --mem=0                     # all of the node memory
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Wing deployment of one controller on one GPU node in a HydroGym image (see ../slurm_image.sh for SIF and BACKEND):
#   POLICY=zero RUN=runs/zero sbatch run_slurm.sbatch
#   POLICY=zero RUN=runs/zero_dt4e-5 ACTIONS=16130 sbatch run_slurm.sbatch
#       (wing.yml: dt 4e-5, 31 steps per action, Ref1's trip; ACTIONS=16130 reaches t = 48)
#   POLICY=zero RUN=runs/zero_ref1 CONF=wing_dt2e-5.yml ACTIONS=16130 sbatch run_slurm.sbatch
#       (the validation time step dt 2e-5, 62 steps per action, also t = 48)
#   POLICY=oc CONF=wing_oc.yml RUN=runs/oc sbatch run_slurm.sbatch
#   POLICY=td3:naca0012_wing_re2e5/td3/seed1998 RUN=runs/td3 sbatch run_slurm.sbatch      (released policy id)
#   POLICY=zero RUN=runs/zero_trip60 TRIP_MODES=60 TRIP_INTERVAL=0.004 CONF=wing_dt8e-5.yml sbatch run_slurm.sbatch
#       (60 trip modes every 0.004)
#   POLICY=td3:<policy zip> RUN=runs/td3 BIND=<directory of the zip> sbatch run_slurm.sbatch
#   POLICY=td3:<policy zip> RUN=runs/td3_utau60 CONF=wing_utau_trip60.yml sbatch run_slurm.sbatch
#       (wing_utau_trip60.yml written first by: python fit_utau.py runs/zero_trip60 --skip 500 --out wing_utau_trip60.yml)
#   BACKEND=HIP POLICY=oc RUN=runs/oc sbatch run_slurm.sbatch      (AMD ROCm image, apptainer --rocm)
# NP NekRS ranks (default 4, on aarch64 8) on the GPUs of the node. On aarch64 nodes the wing needs NP >= 8 (with fewer ranks the
# static Nek5000 arrays per rank exceed the aarch64 relocation range); when NP is larger than the number of GPUs of
# the node, ../gpu_share.sh shares the GPUs among the ranks (GPU_SHARE=1, automatic; GPU_SHARE=0: off), e.g.
# NP=8 on 4 GPUs: 2 ranks per GPU.
# Submit from this directory. The run stops with a chunk shortly before the time limit and the job resubmits
# itself until ACTIONS control actions are done (state.json), to PARTITIONS (default: the partition of this job).
# The Python package and the udf header come from this checkout (PYTHONPATH, NEKRS_UDF_INCLUDES), not from the copy
# in the image.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
HERE=$PWD
REPO=$(cd ../../.. && pwd)
# shellcheck source=../slurm_image.sh
source ../slurm_image.sh       # SIF, BACKEND (CUDA | HIP), GPUFLAG
POLICY="${POLICY:?set POLICY (zero | oc | ub | td3:<zip> | td3:<released policy id>)}"
RUN="${RUN:?set RUN (e.g. runs/zero)}"
ACTIONS="${ACTIONS:-2500}"
NP="${NP:-$([ "$(uname -m)" = aarch64 ] && echo 8 || echo 4)}"   # aarch64: the wing needs NP >= 8
MAX_HOURS="${MAX_HOURS:-7.0}"     # deploy.py wall time (NekRS setup and the first JIT come on top)
CONF="${CONF:-wing.yml}"         # env YAML, e.g. wing_utau_trip60.yml from fit_utau.py
PAR="${PAR:-naca_wing.par}"      # NekRS par file (relative to this directory)
EXCLUDE="${EXCLUDE:-}"           # nodes to exclude on resubmission
BIND="${BIND:-}"                 # further directory to bind into the image, e.g. the directory of a policy zip
GPU_SHARE="${GPU_SHARE:-}"       # 1: share the GPUs among the NekRS ranks (../gpu_share.sh); empty: if NP > GPUs
TRIP_MODES="${TRIP_MODES:-}"     # spanwise modes of the trip (empty: the par's 30)
TRIP_INTERVAL="${TRIP_INTERVAL:-}" # time between new random trip signals (empty: the par's 0.008)
DT="${DT:-}"                     # solver time step (empty: environment.dt of CONF)
EXTRA_ARGS="${EXTRA_ARGS:-}"      # further deploy.py arguments, e.g. "--chunk 100 --alpha 0.5"
GPU_MONITOR="${GPU_MONITOR:-60}"  # seconds between rocm-smi / nvidia-smi samples into RUN/gpu-JOB.log (0: off), see ../perf_report.py

mkdir -p "$RUN"
actions_done() { python3 -c "import json; print(json.load(open('$RUN/state.json'))['actions'])" 2>/dev/null || echo 0; }
before=$(actions_done)
# environment of the container via APPTAINERENV_* (kept by --cleanenv; --env would split the comma-separated
# device list ROCR_VISIBLE_DEVICES=0,1,2,3 or CUDA_VISIBLE_DEVICES=0,1,2,3 that Slurm sets for the exclusive node).
# --cleanenv also keeps the host PATH out: a compiler of a loaded host module would otherwise build the udf.
export APPTAINERENV_PYTHONPATH="$REPO" APPTAINERENV_NEKRS_BACKEND=$BACKEND APPTAINERENV_OMP_NUM_THREADS=1
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
if [[ -z "$GPU_SHARE" ]]; then
    gpus=${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}
    if [[ -n "$gpus" ]]; then
        IFS=, read -ra G <<< "$gpus"; ngpu=${#G[@]}
    else
        ngpu=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)
    fi
    GPU_SHARE=0
    (( ngpu > 0 && NP > ngpu )) && GPU_SHARE=1
fi
if [[ "$BACKEND" == CUDA ]]; then
    SMI="nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv"
else
    SMI="rocm-smi --showuse --showmemuse --showpower --showtemp --csv"
fi
if (( GPU_MONITOR > 0 )); then
    apptainer exec $GPUFLAG "$SIF" bash -c "while true; do echo \"# time \$(date +%s)\"; $SMI; \
        sleep $GPU_MONITOR; done" > "$RUN/gpu-${SLURM_JOB_ID:-local}.log" 2>&1 &
    MONITOR=$!
fi
apptainer exec $GPUFLAG --cleanenv --bind "$REPO" ${BIND:+--bind "$BIND"} \
    "$SIF" bash -c "cd $HERE && NP=$NP GPU_SHARE=$GPU_SHARE RUN=$RUN CONF=$CONF PAR=$PAR TRIP_MODES=$TRIP_MODES TRIP_INTERVAL=$TRIP_INTERVAL DT=$DT ./run.sh --policy $POLICY --actions $ACTIONS --max-hours $MAX_HOURS $EXTRA_ARGS" \
    > "$RUN/deploy-${SLURM_JOB_ID:-local}.log" 2>&1
rc=$?
[[ -n "${MONITOR:-}" ]] && kill "$MONITOR" 2>/dev/null
done=$(actions_done)
echo "exit $rc, $done of $ACTIONS actions ($before at the start of this job)"
# resubmit only if this job made progress (a failing setup must not loop)
if (( done < ACTIONS )) && (( done > before )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "resubmitting"
    PART="${PARTITIONS:-${SLURM_JOB_PARTITION:-}}"
    POLICY="$POLICY" RUN="$RUN" ACTIONS="$ACTIONS" NP="$NP" MAX_HOURS="$MAX_HOURS" EXTRA_ARGS="$EXTRA_ARGS" TRIP_MODES="$TRIP_MODES" TRIP_INTERVAL="$TRIP_INTERVAL" DT="$DT" CONF="$CONF" PAR="$PAR" \
        SIF="$SIF" EXCLUDE="$EXCLUDE" BACKEND="$BACKEND" BIND="$BIND" GPU_SHARE="$GPU_SHARE" \
        sbatch --export=ALL ${PART:+--partition=$PART} ${EXCLUDE:+--exclude=$EXCLUDE} run_slurm.sbatch
elif (( done < ACTIONS )); then
    echo "no progress in this job - not resubmitting, see $RUN/deploy-*.log"
fi
