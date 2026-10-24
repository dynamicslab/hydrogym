#!/usr/bin/env bash
#SBATCH --job-name=hg-pinball
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --account=m-aia
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Train several pinball baselines side by side on one MI300A node, one GPU (APU) each:
#   RUNS="PPO:0 DDPG:0 TD3:0" ENV=Pinball_2D_Re100 sbatch run_aac.sbatch
# Each run gets its own copy of the case workspace (m-AIA writes its output into the cwd) under
# $BASE/runs/<ENV>/<algo>_s<seed>. Runs stop resumably shortly before the 8 h limit; the job
# then resubmits itself until all runs are done (eval_rollout.json written).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # submit from this directory
source ./aac_env.sh
ENV="${ENV:-Pinball_2D_Re100}"
RUNS="${RUNS:-PPO:0 DDPG:0 TD3:0}"
EPISODES="${EPISODES:-500}"
MAX_HOURS="${MAX_HOURS:-7.5}"
NP_MAIA="${NP_MAIA:-1}"
export HF_HUB_OFFLINE=1
TRAIN="${REPO}/examples/maia/paper_baselines/train_pinball.py"

gpu=0; pids=(); todo=0
for run in ${RUNS}; do
    algo="${run%%:*}"; seed="${run##*:}"
    tag="$(echo "${algo}" | tr '[:upper:]' '[:lower:]')_s${seed}"
    ws="${BASE}/runs/${ENV}/${tag}"
    if [[ -f "${ws}/out/eval_rollout.json" ]]; then echo "${tag}: done"; continue; fi
    if [[ ! -f "${ws}/properties_run.toml" ]]; then
        mkdir -p "${ws}"
        cp -a "${BASE}/cases/${ENV}/." "${ws}/"
    fi
    echo "${tag}: GPU ${gpu}, workspace ${ws}"
    ROCR_VISIBLE_DEVICES="${gpu}" in_image "cd ${ws} && mpirun --bind-to none -np 1 python ${TRAIN} --env ${ENV} --algo ${algo} --seed ${seed} --episodes ${EPISODES} --max-hours ${MAX_HOURS} --out out : -np ${NP_MAIA} maia properties_run.toml" > "${ws}/train-${SLURM_JOB_ID:-local}.log" 2>&1 &
    pids+=($!); gpu=$((gpu + 1)); todo=$((todo + 1))
done
(( todo == 0 )) && { echo "All runs done."; exit 0; }

resubmit=0
for pid in "${pids[@]}"; do
    wait "${pid}"; rc=$?
    (( rc == 3 )) && resubmit=1
    (( rc != 0 && rc != 3 )) && echo "WARNING: a run exited with ${rc} - see its train-*.log"
done
if (( resubmit )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Wall-time limit reached - resubmitting"
    ENV="${ENV}" RUNS="${RUNS}" EPISODES="${EPISODES}" MAX_HOURS="${MAX_HOURS}" NP_MAIA="${NP_MAIA}" \
        sbatch --export=ALL run_aac.sbatch   # ($0 is Slurm's spooled copy)
fi
