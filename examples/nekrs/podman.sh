#!/usr/bin/env bash
# Run a command in the HydroGym GPU image (NekRS 26.0 in /opt/nekrs with CUDA-aware Open MPI, nekrs-mpmd on PATH,
# HydroGym in /opt/venvs/hydrogym). This repository is mounted over the image copy (/opt/hydrogym, the editable
# install of the image's venv), so local edits are used; the working directory is mapped to the same place inside.
# MOUNT_REPO=0 uses the copy in the image.
# Usage: GPU=0 examples/nekrs/podman.sh 'nvidia-smi'          (GPU may be a comma-separated list, or empty: no GPU)
#        EXTRA="-v /data:/data" examples/nekrs/podman.sh '...'  (additional run options)
#        ENGINE=docker examples/nekrs/podman.sh '...'          (docker instead of podman)
#        IMG=clagemann/hydrogym-cpu:1.0.2 GPU= NEKRS_BACKEND=SERIAL ...   (CPU image)
GPU=${GPU-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CDIR=/opt/hydrogym
IMG=${IMG:-clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2}
ENGINE=${ENGINE:-podman}
case "$PWD/" in "$REPO"/*) WD=$CDIR${PWD#"$REPO"} ;; *) WD=$CDIR ;; esac
DEVS=()
if [ -n "$GPU" ]; then
  if [ "$ENGINE" = docker ]; then
    DEVS=(--gpus "\"device=$GPU\"")
  else
    for g in ${GPU//,/ }; do DEVS+=(--device "nvidia.com/gpu=$g"); done   # CDI (nvidia-ctk cdi generate)
  fi
fi
VOL=""; [ "${MOUNT_REPO:-1}" = 1 ] && VOL="-v $REPO:$CDIR"
# host CA bundle (TLS-intercepting proxies; Hugging Face downloads fail without it). HOST_CA=0 disables.
CA=/etc/ssl/certs/ca-certificates.crt
if [ "${HOST_CA:-1}" = 1 ] && [ -f $CA ]; then
  VOL="$VOL -v $CA:/etc/ssl/host-ca.crt:ro -e SSL_CERT_FILE=/etc/ssl/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/host-ca.crt"
fi
# The command runs as root in the container (rootless podman maps it to the host user); the image allows mpirun
# as root. NEKRS_BACKEND, NEKRS_MPMD and UCX_TLS are passed on when set (UCX_TLS: keep cuda_copy for GPU-aware MPI).
ENVS=()
for v in NEKRS_BACKEND NEKRS_MPMD UCX_TLS; do [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}"); done
# shellcheck disable=SC2086
exec "$ENGINE" run --rm --user 0 "${DEVS[@]}" --shm-size=8g ${PODMAN_NAME:+--name $PODMAN_NAME} $VOL $EXTRA -w "$WD" \
  -e OMP_NUM_THREADS=1 -e OMPI_MCA_coll_hcoll_enable=0 "${ENVS[@]}" \
  --entrypoint bash "$IMG" -c "ulimit -s unlimited; $1"
# ulimit -s: Nek5000's restart reader (mfi_getv) segfaults with the default 8 MB stack
