#!/usr/bin/env bash
# MPMD launch of the HydroGym controller (rank 0, unchanged hydrogym.nek.NekEnv) and NekRS (ranks 1..NP, one GPU
# each) for the legacy DRL channel coupling in this directory. On the host it re-runs itself in the image through
# ../podman.sh.
# Usage: GPU=3 NP=1 RUN=runs/zero ./run.sh --policy zero --steps 20
#        (arguments go to drl_client.py; --nproc and --run-root are set from NP and RUN, relative to this
#        directory; the solver log of the wall shear goes to $RUN/wall.dat)
HERE=$(cd "$(dirname "$0")" && pwd)
NP=${NP:-1}
PAR=${PAR:-tcf.par}
RUN=${RUN:-runs/test}
cd "$HERE"
# EasyBuild image (/home/easybuild/venvs) or devcontainer (NEKRS_HOME set, features/nekrs); otherwise via podman
if [ ! -d /home/easybuild/venvs ] && [ -z "$NEKRS_HOME" ]; then
  exec "$HERE/../podman.sh" "NP=$NP PAR=$PAR RUN=$RUN ./run.sh $*"
fi
if [ ! -f tcf.re2 ] || [ ! -f rs6tcf0.f00004 ]; then
  echo "[run.sh] tcf.re2 / rs6tcf0.f00004 missing: run ../fetch_case.py first"; exit 1
fi
mkdir -p "$RUN"
if [ -d /home/easybuild/venvs ]; then
  module load Python/3.12.3-GCCcore-13.3.0
  PY=/home/easybuild/venvs/hydrogym_gpu/bin/python
  # mpi4py from its module directory without loading the module, which would swap in the non-CUDA-aware Open MPI
  MPI4PY=/home/easybuild/eb_software/software/mpi4py/4.0.1-gompi-2024a-SystemCUDA/lib/python3.12/site-packages
  export PYTHONPATH=$MPI4PY${PYTHONPATH:+:$PYTHONPATH}
else
  PY=${PY:-/opt/venvs/nekrs/bin/python}
fi
export NEKRS_CACHE_DIR=${NEKRS_CACHE_DIR:-$HERE/.cache}
MPMD=${NEKRS_MPMD:-$(command -v nekrs-mpmd || echo ../mpmd/nekrs-mpmd)}
exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u drl_client.py --nproc $NP --run-root $RUN "$@" : \
  -n $NP env NEKRS_MPMD_COLOR=1 DRL_WALLDAT=$RUN/wall.dat $MPMD --setup $PAR --backend ${NEKRS_BACKEND:-CUDA}
