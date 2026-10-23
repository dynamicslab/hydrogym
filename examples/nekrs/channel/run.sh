#!/usr/bin/env bash
# MPMD launch of the HydroGym NekRS channel: controller (rank 0, SCRIPT, default client.py + channel_re207.yml)
# and NekRS (ranks 1..NP, one GPU each). On the host it re-runs itself in the image through ../podman.sh.
# Usage: GPU=3 NP=1 RUN=runs/zero ./run.sh --policy zero --steps 20
#        GPU=3 NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --episodes 10
#        (arguments go to SCRIPT; --nproc and --run-root are set from NP and RUN, relative to this directory)
HERE=$(cd "$(dirname "$0")" && pwd)
NP=${NP:-1}
PAR=${PAR:-tcf.par}
RUN=${RUN:-runs/test}
SCRIPT=${SCRIPT:-client.py}
cd "$HERE"
# EasyBuild image (/home/easybuild/venvs) or devcontainer (NEKRS_HOME set, features/nekrs); otherwise via podman
if [ ! -d /home/easybuild/venvs ] && [ -z "$NEKRS_HOME" ]; then
  exec "$HERE/../podman.sh" "NP=$NP PAR=$PAR RUN=$RUN SCRIPT=$SCRIPT ./run.sh $*"
fi
if [ ! -f tcf.re2 ] || [ ! -f rs6tcf0.f00004 ]; then
  echo "[run.sh] tcf.re2 / rs6tcf0.f00004 missing: run ../fetch_case.py first"; exit 1
fi
mkdir -p "$RUN"
export NEKRS_CACHE_DIR=${NEKRS_CACHE_DIR:-$HERE/.cache}
# NekRS rebuilds the udf only when the .udf file itself changes, not the header it includes
LIB="$NEKRS_CACHE_DIR/udf/libudf.so"
if [ -f "$LIB" ] && [ "$HERE/../../../hydrogym/nekrs/udf/hydrogym_nekrs.hpp" -nt "$LIB" ]; then
  echo "[run.sh] hydrogym_nekrs.hpp changed, rebuilding the udf"; rm -rf "$NEKRS_CACHE_DIR/udf"
fi
if [ -d /home/easybuild/venvs ]; then
  module load Python/3.12.3-GCCcore-13.3.0
  PY=/home/easybuild/venvs/hydrogym_gpu/bin/python
  # mpi4py from its module directory without loading the module, which would swap in the non-CUDA-aware Open MPI
  MPI4PY=/home/easybuild/eb_software/software/mpi4py/4.0.1-gompi-2024a-SystemCUDA/lib/python3.12/site-packages
  export PYTHONPATH=$MPI4PY${PYTHONPATH:+:$PYTHONPATH}
else
  PY=${PY:-/opt/venvs/nekrs/bin/python}
fi
MPMD=${NEKRS_MPMD:-$(command -v nekrs-mpmd || echo ../mpmd/nekrs-mpmd)}
exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u $SCRIPT --nproc $NP --run-root $RUN "$@" : \
  -n $NP env NEKRS_MPMD_COLOR=1 $MPMD --setup $PAR --backend ${NEKRS_BACKEND:-CUDA}
