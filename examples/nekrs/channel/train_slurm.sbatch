#!/usr/bin/env bash
#SBATCH --job-name=hg-td3
# add "#SBATCH --partition=<partition>" and "#SBATCH --account=<project>" here if the cluster requires them
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5         # mpirun slots: train_td3.py + NSIM x NP NekRS ranks (one GPU each)
#SBATCH --exclusive                 # the whole node: all GPUs and CPU cores
#SBATCH --mem=0                     # all of the node memory
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# TD3 training on the channel on one GPU node in a HydroGym image (see ../slurm_image.sh for SIF and BACKEND),
# NSIM simulations (one GPU each) feeding one policy:
#   RUN=runs/td3_x4 NSIM=4 EPISODES=100 sbatch train_slurm.sbatch
#   RUN=runs/td3_s0 NSIM=1 EXTRA_ARGS="--seed 20010603 --buffer-size 20000000" sbatch train_slurm.sbatch
#   BACKEND=HIP RUN=runs/td3_x4 sbatch train_slurm.sbatch      (AMD ROCm image, apptainer --rocm)
# Submit from this directory. Every episode is checkpointed (RUN/checkpoints/latest); the training stops after
# the last episode that fits into MAX_HOURS and the job resubmits itself until EPISODES episodes per simulation
# are done, to PARTITIONS (default: the partition of this job). The Python package and the udf header come from
# this checkout.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
HERE=$PWD
REPO=$(cd ../../.. && pwd)
# shellcheck source=../slurm_image.sh
source ../slurm_image.sh       # SIF, BACKEND (CUDA | HIP), GPUFLAG
RUN="${RUN:?set RUN (e.g. runs/td3_x4)}"
CONF="${CONF:-channel_paper.yml}"
NSIM="${NSIM:-4}"
NP="${NP:-1}"
EPISODES="${EPISODES:-100}"
MAX_HOURS="${MAX_HOURS:-7.5}"     # wall time budget of train_td3.py in this job (NekRS setup included)
EXTRA_ARGS="${EXTRA_ARGS:-}"      # further train_td3.py arguments, e.g. "--seed 1 --buffer-size 80000000"
BIND="${BIND:-}"                 # further directory to bind into the image
GPU_MONITOR="${GPU_MONITOR:-60}"  # seconds between nvidia-smi / rocm-smi samples into RUN/gpu-JOB.log (0: off), see ../perf_report.py

mkdir -p "$RUN"
STATE=$RUN/checkpoints/latest/train_state.json
progress() { python3 -c "import json; print(json.load(open('$STATE'))['actions_per_sim'])" 2>/dev/null || echo 0; }
finished() { python3 -c "import json; print(int(json.load(open('$STATE'))['done']))" 2>/dev/null || echo 0; }
before=$(progress)
# environment of the container via APPTAINERENV_* (kept by --cleanenv; --env would split the comma-separated
# device list CUDA_VISIBLE_DEVICES=0,1,2,3 or ROCR_VISIBLE_DEVICES=0,1,2,3 that Slurm sets for the exclusive node)
export APPTAINERENV_PYTHONPATH="$REPO" APPTAINERENV_NEKRS_BACKEND=$BACKEND APPTAINERENV_OMP_NUM_THREADS=1
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
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
    "$SIF" bash -c "cd $HERE && ulimit -s unlimited && NP=$NP NSIM=$NSIM RUN=$RUN SCRIPT=train_td3.py ./run.sh \
        --conf $CONF --episodes $EPISODES --max-hours $MAX_HOURS $EXTRA_ARGS" \
    >> "$RUN/train-${SLURM_JOB_ID:-local}.log" 2>&1
rc=$?
[[ -n "${MONITOR:-}" ]] && kill "$MONITOR" 2>/dev/null
done=$(progress)
echo "exit $rc, $done actions per simulation ($before at the start of this job)"
# resubmit only if this job made progress (a failing setup must not loop)
if (( $(finished) == 0 )) && (( done > before )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "resubmitting"
    PART="${PARTITIONS:-${SLURM_JOB_PARTITION:-}}"
    RUN="$RUN" CONF="$CONF" NSIM="$NSIM" NP="$NP" EPISODES="$EPISODES" MAX_HOURS="$MAX_HOURS" EXTRA_ARGS="$EXTRA_ARGS" \
        SIF="$SIF" BACKEND="$BACKEND" BIND="$BIND" sbatch --export=ALL ${PART:+--partition=$PART} train_slurm.sbatch
elif (( $(finished) == 0 )); then
    echo "no progress in this job - not resubmitting, see $RUN/train-*.log"
fi
