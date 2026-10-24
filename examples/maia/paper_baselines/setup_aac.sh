#!/usr/bin/env bash
# One-time setup on the AAC login node (needs network): pull the HydroGym ROCm image and
# download the pinball cases into the HF cache.
# Usage: REPO=~/hydrogym BASE=~/hydrogym-baselines ./setup_aac.sh
set -euo pipefail
source "$(dirname "$0")/aac_env.sh"
mkdir -p "${BASE}"

if [[ ! -f "${SIF}" ]]; then
    export APPTAINER_TMPDIR="/tmp/${USER}/apptainer-tmp" APPTAINER_CACHEDIR="/tmp/${USER}/apptainer-cache"
    mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
    apptainer pull "${SIF}" "${IMAGE}"
fi

in_image "
set -e
which python maia
python -c 'import mpi4py, torch, stable_baselines3, hydrogym; print(\"mpi4py\", mpi4py.__version__); print(\"torch\", torch.__version__, \"sb3\", stable_baselines3.__version__)'
for e in Pinball_2D_Re100 Pinball_3D_Re150; do
    python ${REPO}/examples/maia/getting_started/prepare_workspace.py --env \$e --work-dir ${BASE}/cases/\$e
done
"
echo "Setup done: ${BASE}"
