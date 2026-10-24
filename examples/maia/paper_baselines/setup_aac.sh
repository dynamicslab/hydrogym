#!/usr/bin/env bash
# One-time setup on the AAC login node (needs network): pull the m-AIA ROCm image, build the
# Python venv inside it and download the pinball cases into the HF cache.
# Usage: REPO=~/hydrogym BASE=~/hydrogym-baselines ./setup_aac.sh
set -euo pipefail
source "$(dirname "$0")/aac_env.sh"
mkdir -p "${BASE}"

if [[ ! -f "${SIF}" ]]; then
    export APPTAINER_TMPDIR="/tmp/${USER}/apptainer-tmp" APPTAINER_CACHEDIR="/tmp/${USER}/apptainer-cache"
    mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
    apptainer pull "${SIF}" "${IMAGE}"
fi

# The module's Python already has mpi4py built against the image's Open MPI (the one m-AIA
# uses) - --system-site-packages keeps it; a pip-installed mpi4py wheel would bring its own MPI.
# CPU torch: the policy networks are small and the GPU belongs to m-AIA.
in_image "
set -e
which python; python -c 'import mpi4py; print(\"mpi4py\", mpi4py.__version__, mpi4py.__file__)'
[[ -d ${VENV} ]] || python -m venv --system-site-packages ${VENV}
source ${VENV}/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '${REPO}[maia]'
python -c 'import mpi4py, torch, stable_baselines3, hydrogym; print(\"mpi4py\", mpi4py.__file__); print(\"torch\", torch.__version__, \"sb3\", stable_baselines3.__version__)'
which maia
for e in Pinball_2D_Re100 Pinball_3D_Re150; do
    python ${REPO}/examples/maia/getting_started/prepare_workspace.py --env \$e --work-dir ${BASE}/cases/\$e
done
"
echo "Setup done: ${BASE}"
