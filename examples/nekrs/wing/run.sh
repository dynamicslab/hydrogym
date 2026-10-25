#!/usr/bin/env bash
# MPMD launch of the NACA0012 wing deployment: deploy.py (rank 0, SCRIPT) and NekRS (ranks 1..NP, one GPU each),
# both in the run directory RUN, which holds the case links, the frozen far field, the trip origin, the chunk
# fields and state.json. A relaunch with the same RUN continues from the last chunk. On the host it re-runs
# itself in the image through ../podman.sh.
# Usage: GPU=1,2,3,4 NP=4 RUN=runs/zero ./run.sh --policy zero --actions 2500
#        GPU=1,2,3,4 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$PWD/../channel/runs/paper_s0/td3_final.zip
#        GPU=1 NP=1 RUN=runs/smoke PAR=naca_wing_smoke.par CONF=wing_smoke.yml ./run.sh --policy oc --actions 4
#        (arguments go to SCRIPT; --nproc, --run-root and --conf are set here; START: initial field, default
#        restart/rs8naca_wing0.f00008 from prepare_case.py)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
NP=${NP:-4}
PAR=${PAR:-naca_wing.par}
CONF=${CONF:-wing.yml}
RUN=${RUN:-runs/test}
SCRIPT=${SCRIPT:-deploy.py}
START=${START:-$HERE/restart/rs8naca_wing0.f00008}
cd "$HERE"
if [ ! -d /home/easybuild/venvs ] && [ -z "$NEKRS_HOME" ] && [ ! -d /opt/venvs/nekrs ]; then
  # the restart may be a symlink to a directory outside the repository: mount that too
  EXTRA_START=""
  if [ -e "$START" ]; then
    D=$(dirname "$(readlink -f "$START")")
    case "$D/" in "$REPO"/*) ;; *) EXTRA_START="-v $D:$D:ro" ;; esac
  fi
  EXTRA="$EXTRA $EXTRA_START" exec "$HERE/../podman.sh" \
    "NP=$NP PAR=$PAR CONF=$CONF RUN=$RUN SCRIPT=$SCRIPT START=$START ./run.sh $*"
fi

mkdir -p "$RUN"
RUNDIR=$(cd "$RUN" && pwd)
ln -sf "$REPO/third_party/nek5000/cases/naca0012_200k/naca_wing.re2" "$RUNDIR/naca_wing.re2"
ln -sf "$HERE/naca_wing.usr" "$RUNDIR/naca_wing.usr"
ln -sf "$HERE/naca_wing.udf" "$RUNDIR/naca_wing.udf"
cp "$HERE/$PAR" "$RUNDIR/naca_wing.par"
# initial field: the last chunk of an earlier launch, else START
LAST=""
if [ -f "$RUNDIR/state.json" ]; then
  LAST=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['chunks']; print(c[-1]['file'] if c else '')" \
    "$RUNDIR/state.json")
fi
if [ -n "$LAST" ]; then
  ln -sf "$LAST" "$RUNDIR/start0.f00001"
  echo "[run.sh] continuing from $LAST"
  # (the smoke par starts from a uniform field: read the chunk there too)
  grep -q '^ *startFrom' "$RUNDIR/naca_wing.par" ||
    sed -i '/^\[GENERAL\]/a startFrom = "start0.f00001+U+P"' "$RUNDIR/naca_wing.par"
elif grep -q '^ *startFrom' "$RUNDIR/naca_wing.par"; then
  if [ ! -e "$START" ]; then
    echo "[run.sh] $START missing: run ./prepare_case.py first (or set START)"; exit 1
  fi
  ln -sf "$(readlink -f "$START")" "$RUNDIR/start0.f00001"
fi

export NEKRS_UDF_INCLUDES=$REPO/hydrogym/nekrs/udf
export NEKRS_CACHE_DIR=${NEKRS_CACHE_DIR:-$RUNDIR/.cache}
# NekRS rebuilds the udf only when the .udf file itself changes, not the header it includes
LIB="$NEKRS_CACHE_DIR/udf/libudf.so"
if [ -f "$LIB" ] && [ "$NEKRS_UDF_INCLUDES/hydrogym_nekrs.hpp" -nt "$LIB" ]; then
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
MPMD=${NEKRS_MPMD:-$(command -v nekrs-mpmd || echo "$HERE/../mpmd/nekrs-mpmd")}
SCRIPT=$(readlink -f "$SCRIPT")
cd "$RUNDIR"
exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u "$SCRIPT" --conf "$HERE/$CONF" --nproc $NP --run-root . "$@" : \
  -n $NP env NEKRS_MPMD_COLOR=1 $MPMD --setup naca_wing.par --backend ${NEKRS_BACKEND:-CUDA}
