#!/bin/bash
# add #SBATCH --partition/--account if your cluster requires them
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Training chain for the PPO / GPPO runs: up to 4 runs (one per GPU), <= 1 h per job. Every run checkpoints after
# each update and stops cleanly after 50 min (--max-minutes); the job then resubmits itself until all its runs have
# <algo>_DONE (at most 5 attempts). A 100-update run needs 2 jobs.
# usage: KGP_RUN_DIR=/path/to/run_dir sbatch --export=ALL --job-name=kgp_tr_s30 train_chain.sbatch 1 \
#            ppo_mb:30 ppo_ro:30 gppo_b0:30 gppo_b2:30
# (submit_all.sh submits one chain per seed.)
R=${KGP_RUN_DIR:?set KGP_RUN_DIR}
EX=${KGP_EXAMPLE_DIR:?set KGP_EXAMPLE_DIR (examples/jax/kolmogorov_gppo)}
PY=${KGP_PYTHON:-python}
ATT=$1; shift
export PYTHONPATH=$(cd "$EX/../../.." && pwd) XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$EX" || exit 1
i=0
for spec in "$@"; do
  v=${spec%%:*}; s=${spec##*:}; algo=${v%%_*}
  case $v in
    ppo_mb) args="ppo.py";; ppo_ro) args="ppo.py --adv-norm rollout";;
    gppo_b2) args="gppo.py --beta 2.0";; gppo_b0) args="gppo.py --beta 0";;
    *) echo "bad variant $v"; exit 2;;
  esac
  od=$R/runs/$v/s$s; mkdir -p "$od"
  if [ ! -f "$od/${algo}_DONE" ]; then
    echo "$(date -Is) job $SLURM_JOB_ID attempt $ATT gpu $i: $args --seed $s" >> "$R/RUNS.log"
    CUDA_VISIBLE_DEVICES=$i $PY $args --seed $s --out-dir "$od" --num-updates 100 --resume --max-minutes 50 \
      >> "$od/train.log" 2>&1 &
  fi
  i=$((i+1))
done
wait
left=0
for spec in "$@"; do v=${spec%%:*}; s=${spec##*:}; algo=${v%%_*}; [ -f "$R/runs/$v/s$s/${algo}_DONE" ] || left=$((left+1)); done
if [ $left -gt 0 ] && [ "$ATT" -lt 5 ]; then
  sbatch --export=ALL --job-name="$SLURM_JOB_NAME" "$EX/slurm/train_chain.sbatch" $((ATT+1)) "$@"
fi
