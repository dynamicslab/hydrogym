#!/bin/bash
# Submit the PPO / GPPO runs on Slurm: one training chain per seed (4 variants x 1 seed per 4-GPU node) and the
# main evaluation chain. Add the #SBATCH partition/account lines your cluster requires first. After training, run
#   sbatch --export=ALL --job-name=kgp_ev_tr eval_chain.sbatch 1 trained
# and then:  python ../analyse.py $KGP_RUN_DIR $KGP_RUN_DIR/analysis
# usage: KGP_RUN_DIR=/path/to/run_dir KGP_PYTHON=/path/to/venv/bin/python bash submit_all.sh [seeds]
set -e
export KGP_EXAMPLE_DIR=$(cd "$(dirname "$0")/.." && pwd)
: "${KGP_RUN_DIR:?set KGP_RUN_DIR}"
mkdir -p "$KGP_RUN_DIR"
cd "$KGP_RUN_DIR"
for s in ${1:-30 31 32 33 34}; do
  sbatch --export=ALL --job-name=kgp_tr_s$s "$KGP_EXAMPLE_DIR/slurm/train_chain.sbatch" 1 \
    ppo_mb:$s ppo_ro:$s gppo_b0:$s gppo_b2:$s
done
sbatch --export=ALL --job-name=kgp_ev_main "$KGP_EXAMPLE_DIR/slurm/eval_chain.sbatch" 1 main
