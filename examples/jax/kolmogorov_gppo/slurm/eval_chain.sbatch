#!/bin/bash
# add #SBATCH --partition/--account if your cluster requires them
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Evaluation chain: 4 evaluate.py workers (one per GPU) share the task queue in $KGP_RUN_DIR/eval; resubmits itself
# until every task is done (at most 6 attempts). Mode "main": uncontrolled statistics, bank, released policies,
# baselines (~25 min on 4 GPUs); mode "trained": also every run in $KGP_RUN_DIR/runs (after training).
# usage: KGP_RUN_DIR=... KGP_EXAMPLE_DIR=... sbatch --export=ALL --job-name=kgp_ev eval_chain.sbatch 1 main|trained
R=${KGP_RUN_DIR:?set KGP_RUN_DIR}
EX=${KGP_EXAMPLE_DIR:?set KGP_EXAMPLE_DIR}
PY=${KGP_PYTHON:-python}
ATT=$1; MODE=$2
export PYTHONPATH=$(cd "$EX/../../.." && pwd) XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=$R/eval; mkdir -p "$OUT"; rm -rf "$OUT/locks"; cd "$EX" || exit 1
EXTRA=""; [ "$MODE" = trained ] && EXTRA="--with-trained --runs-root $R/runs"
T0=$(date +%s)
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY evaluate.py --out "$OUT" --policy-root "$EX/../../../policies/kolmogorov_re40" $EXTRA \
     --job-start $T0 --minutes 52 >> "$OUT/worker_g$g.log" 2>&1 &
  sleep 20
done
wait
st=$($PY evaluate.py --out "$OUT" --policy-root x $EXTRA --status 2>/dev/null | tail -1)
echo "$(date -Is) job $SLURM_JOB_ID attempt $ATT: $st"
if [ "$st" != complete ] && [ "$ATT" -lt 6 ]; then
  sbatch --export=ALL --job-name="$SLURM_JOB_NAME" "$EX/slurm/eval_chain.sbatch" $((ATT+1)) $MODE
fi
