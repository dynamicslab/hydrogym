#!/usr/bin/env bash
# MPMD launch of the HydroGym NekRS channel: controller (rank 0, SCRIPT, default client.py + channel_re207.yml)
# and NekRS (ranks 1..NP, one GPU each). On the host it re-runs itself in the image through ../podman.sh.
# Usage: GPU=0 NP=1 RUN=runs/zero ./run.sh --policy zero --steps 20
#        GPU=0 NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --episodes 10
#        GPU=0,1 NSIM=2 RUN=runs/td3_x2 SCRIPT=train_td3.py ./run.sh --episodes 10
#        (arguments go to SCRIPT; --nproc and --run-root are set from NP and RUN, relative to this directory)
# NSIM > 1: NSIM independent NekRS instances of NP ranks (MPMD colours 1..NSIM) for one controller (--sims NSIM,
# hydrogym.nekrs.multisim). Instance k gets GPUs k*NP .. k*NP+NP-1 of the visible ones, its own kernel cache
# (NEKRS_CACHE_DIR/simk, so the instances do not JIT-compile into the same directory) and the log RUN/nekrs_simk.log.
HERE=$(cd "$(dirname "$0")" && pwd)
NP=${NP:-1}
NSIM=${NSIM:-1}
PAR=${PAR:-tcf.par}
RUN=${RUN:-runs/test}
SCRIPT=${SCRIPT:-client.py}
cd "$HERE"
# HydroGym image or devcontainer (NEKRS_HOME set; /opt/nekrs, features/nekrs); otherwise in the image via podman
if [ -z "$NEKRS_HOME" ] && [ ! -d /opt/venvs/nekrs ]; then
  exec "$HERE/../podman.sh" "NP=$NP NSIM=$NSIM PAR=$PAR RUN=$RUN SCRIPT=$SCRIPT ./run.sh $*"
fi
if [ ! -f tcf.re2 ] || [ ! -f rs6tcf0.f00004 ]; then
  echo "[run.sh] tcf.re2 / rs6tcf0.f00004 missing: run ../fetch_case.py first"; exit 1
fi
mkdir -p "$RUN"
export NEKRS_CACHE_DIR=${NEKRS_CACHE_DIR:-$HERE/.cache}
CACHES=$NEKRS_CACHE_DIR
[ "$NSIM" -gt 1 ] && CACHES=$(for k in $(seq 0 $((NSIM - 1))); do echo "$NEKRS_CACHE_DIR/sim$k"; done)
# NekRS rebuilds the udf only when the .udf file itself changes, not the header it includes
for C in $CACHES; do
  LIB="$C/udf/libudf.so"
  if [ -f "$LIB" ] && [ "$HERE/../../../hydrogym/nekrs/udf/hydrogym_nekrs.hpp" -nt "$LIB" ]; then
    echo "[run.sh] hydrogym_nekrs.hpp changed, rebuilding the udf in $C"; rm -rf "$C/udf"
  fi
done
# Python of the HydroGym images (/opt/venvs/hydrogym, also reachable as /opt/venvs/nekrs); PY=... for another one
PY=${PY:-/opt/venvs/nekrs/bin/python}
MPMD=${NEKRS_MPMD:-$(command -v nekrs-mpmd || echo ../mpmd/nekrs-mpmd)}
BACKEND=${NEKRS_BACKEND:-CUDA}
if [ "$NSIM" -le 1 ] && [ -n "$SLURM_JOB_ID" ] && ! command -v mpirun > /dev/null; then
  # Slurm cluster whose MPI has no mpirun (e.g. some Open MPI installations): the same MPMD launch with srun --multi-prog
  printf '0 %s -u %s --nproc %s --run-root %s %s\n1-%s env NEKRS_MPMD_COLOR=1 %s --setup %s --backend %s\n' \
    "$PY" "$SCRIPT" "$NP" "$RUN" "$*" "$NP" "$MPMD" "$PAR" "$BACKEND" > "$RUN/mpmd.conf"
  exec srun -n $((NP + 1)) --cpu-bind=none --gpu-bind=none --multi-prog "$RUN/mpmd.conf"
fi
if [ "$NSIM" -le 1 ]; then
  exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u $SCRIPT --nproc $NP --run-root $RUN "$@" : \
    -n $NP env NEKRS_MPMD_COLOR=1 $MPMD --setup $PAR --backend $BACKEND
fi
# GPUs of the instances: entries of CUDA_VISIBLE_DEVICES (CUDA) or indices into the visible devices (HIP; on some
# clusters Slurm restricts ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES selects among those)
if [ "$BACKEND" = HIP ]; then VIS=HIP_VISIBLE_DEVICES; else VIS=CUDA_VISIBLE_DEVICES; fi
if [ "$BACKEND" = CUDA ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra DEV <<< "$CUDA_VISIBLE_DEVICES"
else
  DEV=($(seq 0 $((NSIM * NP - 1))))
fi
if [ ${#DEV[@]} -lt $((NSIM * NP)) ]; then
  echo "[run.sh] NSIM=$NSIM x NP=$NP needs $((NSIM * NP)) GPUs, visible: ${DEV[*]}"; exit 1
fi
APPS=""
for k in $(seq 0 $((NSIM - 1))); do
  G=$(IFS=,; echo "${DEV[*]:$((k * NP)):$NP}")
  APPS="$APPS : -n $NP env NEKRS_MPMD_COLOR=$((k + 1)) $VIS=$G NEKRS_CACHE_DIR=$NEKRS_CACHE_DIR/sim$k"
  APPS="$APPS $MPMD --setup $PAR --backend $BACKEND --output $RUN/nekrs_sim$k.log"
  echo "[run.sh] simulation $k: GPU(s) $VIS=$G, log $RUN/nekrs_sim$k.log"
done
exec mpirun --allow-run-as-root --bind-to none -n 1 $PY -u $SCRIPT --nproc $NP --sims $NSIM --run-root $RUN "$@" $APPS
