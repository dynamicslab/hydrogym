#!/usr/bin/env bash
# Run a command in the HydroGym NekRS image (CUDA-aware Open MPI, NekRS module loaded). This repository is
# mounted over the image copy (/home/easybuild/hydrogym, editable install in the venv), so local edits are used;
# the working directory is mapped to the same place inside. MOUNT_REPO=0 uses the copy baked into the image.
# Usage: GPU=3 examples/nekrs/podman.sh 'nvidia-smi'          (GPU may be a comma-separated list)
#        EXTRA="-v /data:/data" examples/nekrs/podman.sh '...'  (additional podman options)
GPU=${GPU:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CDIR=/home/easybuild/hydrogym
IMG=${IMG:-localhost/hydrogym-nekrs:26.0-gpumpi-turing_ampere-hgnekrs}
MOD=${MOD:-NekRS/26.0-gompi-2024a-CUDAaware}
case "$PWD/" in "$REPO"/*) WD=$CDIR${PWD#"$REPO"} ;; *) WD=$CDIR ;; esac
DEVS=""; for g in ${GPU//,/ }; do DEVS="$DEVS --device nvidia.com/gpu=$g"; done
VOL=""; [ "${MOUNT_REPO:-1}" = 1 ] && VOL="-v $REPO:$CDIR"
# host CA bundle (TLS-intercepting proxies; Hugging Face downloads fail without it). HOST_CA=0 disables.
CA=/etc/ssl/certs/ca-certificates.crt
if [ "${HOST_CA:-1}" = 1 ] && [ -f $CA ]; then
  VOL="$VOL -v $CA:/etc/ssl/host-ca.crt:ro -e SSL_CERT_FILE=/etc/ssl/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/host-ca.crt"
fi
# Container root maps to the host user in the default user namespace (--userns=keep-id would create a full
# ID-mapped copy of the 74 GB image); mpirun then needs the allow-run-as-root settings.
exec podman run --rm --user 0 -e HOME=/home/easybuild -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  $DEVS --shm-size=8g ${PODMAN_NAME:+--name $PODMAN_NAME} $VOL $EXTRA -w "$WD" \
  -e OMP_NUM_THREADS=1 -e OMPI_MCA_coll_hcoll_enable=0 \
  --entrypoint bash $IMG -lc "ulimit -s unlimited; module purge >/dev/null 2>&1; module load $MOD; $1"
# ulimit -s: Nek5000's restart reader (mfi_getv) segfaults with the default 8 MB stack
