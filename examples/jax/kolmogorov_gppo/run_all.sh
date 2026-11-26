#!/bin/bash
# PPO / GPPO runs, local (one GPU or CPU): train every variant and seed, evaluate, analyse.
# usage: bash run_all.sh [run dir, default ./kgp_runs]
#   SEEDS="30 31 32 33 34" VARIANTS="ppo_mb ppo_ro gppo_b0 gppo_b2" bash run_all.sh /path/to/run_dir
# Each run is resumable (re-running this script continues from <out-dir>/<algo>_ckpt.pkl; finished runs are skipped).
# Cost: one GPU about 40 s per update (~70 min per 100-update run); CPU ~2-3 min per update.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${1:-$PWD/kgp_runs}
SEEDS=${SEEDS:-"30 31 32 33 34"}
VARIANTS=${VARIANTS:-"ppo_mb ppo_ro gppo_b0 gppo_b2"}
export PYTHONPATH=$(cd "$HERE/../../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PY:-python}
mkdir -p "$RUN/runs" "$RUN/eval"

variant_args() {  # the four variants of the comparison (README, "Variants")
  case $1 in
    ppo_mb)  echo "ppo.py";;                       # PPO (minibatch advantage normalisation)
    ppo_ro)  echo "ppo.py --adv-norm rollout";;    # control: PPO with GPPO's whole-rollout normalisation
    gppo_b0) echo "gppo.py --beta 0";;              # control: GPPO without the gradient term
    gppo_b2) echo "gppo.py --beta 2.0";;            # GPPO
    *) echo "unknown variant $1" >&2; exit 2;;
  esac
}

for s in $SEEDS; do
  for v in $VARIANTS; do
    od="$RUN/runs/$v/s$s"; algo=${v%%_*}
    [ -f "$od/${algo}_DONE" ] && continue
    mkdir -p "$od"
    (cd "$HERE" && $PY $(variant_args $v) --seed $s --out-dir "$od" --num-updates 100 --resume >> "$od/train.log" 2>&1)
  done
done
(cd "$HERE" && $PY evaluate.py --out "$RUN/eval" --policy-root "$HERE/../../../policies/kolmogorov_re40" \
   --with-trained --runs-root "$RUN/runs" --minutes 100000)
$PY "$HERE/analyse.py" "$RUN" "$RUN/analysis"
echo "results: $RUN/analysis/results.json"
