#!/usr/bin/env bash
# MPMD launch of the NACA0012 wing deployment: deploy.py (rank 0, SCRIPT) and NekRS (ranks 1..NP, one GPU each),
# both in the run directory RUN, which holds the case links, the frozen far field, the trip origin, the chunk
# fields and state.json. A relaunch with the same RUN continues from the last chunk. On the host it re-runs
# itself in the image through ../podman.sh.
# Usage: GPU=0,1,2,3 NP=4 RUN=runs/zero ./run.sh --policy zero --actions 2500
#        GPU=0,1,2,3 NP=4 RUN=runs/td3 ./run.sh --policy td3:naca0012_wing_re2e5/td3/seed1998   (released policy id)
#        GPU=0,1,2,3 NP=4 RUN=runs/td3_s0 ./run.sh --policy td3:$PWD/../channel/runs/td3_s0/td3_final.zip
#        GPU=0,1,2,3 NP=4 RUN=runs/oc CONF=wing_oc.yml ./run.sh --policy oc
#        GPU=0 NP=1 RUN=runs/smoke PAR=naca_wing_smoke.par CONF=wing_smoke.yml ./run.sh --policy oc --actions 4
#        trip with 60 modes every 0.004 instead of 30 every 0.008 (validation: C_D -6.4 %):
#        GPU=0,1,2,3 NP=4 RUN=runs/zero_trip60 TRIP_MODES=60 TRIP_INTERVAL=0.004 CONF=wing_dt8e-5.yml \
#          ./run.sh --policy zero
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
# HydroGym image or devcontainer (NEKRS_HOME set; /opt/nekrs, features/nekrs); otherwise in the image via podman
if [ -z "$NEKRS_HOME" ] && [ ! -d /opt/venvs/nekrs ]; then
  # the restart may be a symlink to a directory outside the repository: mount that too
  EXTRA_START=""
  if [ -e "$START" ]; then
    D=$(dirname "$(readlink -f "$START")")
    case "$D/" in "$REPO"/*) ;; *) EXTRA_START="-v $D:$D:ro" ;; esac
  fi
  EXTRA="$EXTRA $EXTRA_START" exec "$HERE/../podman.sh" \
    "NP=$NP PAR=$PAR CONF=$CONF RUN=$RUN SCRIPT=$SCRIPT START=$START GPU_SHARE=${GPU_SHARE:-0} TRIP_MODES=$TRIP_MODES TRIP_INTERVAL=$TRIP_INTERVAL DT=$DT ./run.sh $*"
fi

# the mesh is a Git LFS file: a checkout without git-lfs has only the pointer text
RE2="$REPO/third_party/nek5000/cases/naca0012_200k/naca_wing.re2"
if head -c 8 "$RE2" | grep -q '^version'; then
  echo "[run.sh] $RE2 is a Git LFS pointer: run 'git lfs pull' in $REPO"; exit 1
fi
# Python of the HydroGym images (/opt/venvs/hydrogym, also reachable as /opt/venvs/nekrs); PY=... for another one
PY=${PY:-/opt/venvs/nekrs/bin/python}
mkdir -p "$RUN"
RUNDIR=$(cd "$RUN" && pwd)
ln -sf "$REPO/third_party/nek5000/cases/naca0012_200k/naca_wing.re2" "$RUNDIR/naca_wing.re2"
ln -sf "$HERE/naca_wing.usr" "$RUNDIR/naca_wing.usr"
ln -sf "$HERE/naca_wing.udf" "$RUNDIR/naca_wing.udf"
cp "$HERE/$PAR" "$RUNDIR/naca_wing.par"
# TRIP_MODES: number of spanwise modes of the trip signal (default of the par: 30, as naca_wing.wall)
[ -n "$TRIP_MODES" ] && sed -i "s/^tripModes *=.*/tripModes = $TRIP_MODES/" "$RUNDIR/naca_wing.par"
# TRIP_INTERVAL: time between new random trip signals (default of the par: 0.008, as naca_wing.wall)
[ -n "$TRIP_INTERVAL" ] && sed -i "s/^tripInterval *=.*/tripInterval = $TRIP_INTERVAL/" "$RUNDIR/naca_wing.par"
# DT: solver time step; default environment.dt of CONF (following its base: files), else the par (4e-5). A relaunch
# must use the dt of the first launch (the env stops if environment.dt and the solver dt differ).
if [ -z "$DT" ]; then
  DT=$("$PY" "$HERE/conf_dt.py" "$HERE/$CONF") || { echo "[run.sh] cannot read environment.dt of $CONF"; exit 1; }
fi
[ -n "$DT" ] && sed -i "s/^dt *=.*/dt = $DT/" "$RUNDIR/naca_wing.par"
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
MPMD=${NEKRS_MPMD:-$(command -v nekrs-mpmd || echo "$HERE/../mpmd/nekrs-mpmd")}
# GPU_SHARE=1: more NekRS ranks per node than GPUs (e.g. NP=8 on a node with 4 GPUs), see ../gpu_share.sh
[ "${GPU_SHARE:-0}" = 1 ] && MPMD="$HERE/../gpu_share.sh $MPMD"
# each NekRS rank on the NUMA domain of its GPU (HG_BIND=0: off), see ../bind_numa.sh
SCRIPT=$(readlink -f "$SCRIPT")
cd "$RUNDIR"
if [ -n "$SLURM_JOB_ID" ] && ! command -v mpirun > /dev/null; then
  # Slurm cluster whose MPI has no mpirun (e.g. some Open MPI installations): the same MPMD launch with srun --multi-prog
  printf '0 %s -u %s --conf %s --nproc %s --run-root . %s\n1-%s env NEKRS_MPMD_COLOR=1 %s %s --setup naca_wing.par --backend %s\n' \
    "$PY" "$SCRIPT" "$HERE/$CONF" "$NP" "$*" "$NP" "$HERE/../bind_numa.sh" "$MPMD" "${NEKRS_BACKEND:-CUDA}" > mpmd.conf
  cat mpmd.conf
  exec srun -n $((NP + 1)) --cpu-bind=none --gpu-bind=none --multi-prog mpmd.conf
fi
exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u "$SCRIPT" --conf "$HERE/$CONF" --nproc $NP --run-root . "$@" : \
  -n $NP env NEKRS_MPMD_COLOR=1 "$HERE/../bind_numa.sh" $MPMD --setup naca_wing.par --backend ${NEKRS_BACKEND:-CUDA}
