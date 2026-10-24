# Shared settings for setup_aac.sh / run_aac.sbatch (sourced, not run).
# m-AIA (hipstdpar, gfx942) comes from the colleague image; Python side is a venv on the shared
# file system, built inside that image on top of its m-AIA module's Python + mpi4py.
REPO="${REPO:-$HOME/hydrogym}"                  # hydrogym checkout (kl/nekrs or later)
BASE="${BASE:-$HOME/hydrogym-baselines}"        # image, venv, HF cache, runs
SIF="${SIF:-${BASE}/hydrogym-maia-rocm.sif}"
IMAGE="${IMAGE:-docker://docker.io/clagemann/hydrogym-rocm-6.3.3:v0.1}"
MAIA_MODULE="${MAIA_MODULE:-MAIA/1.0-HIPCC-7.1-OMPI-5.0.5-SystemROCm-6.3.3}"
VENV="${BASE}/venv"
export HF_HOME="${BASE}/hf_cache"

# Run "$@" (a bash command string) in the image with the m-AIA module and the venv loaded.
in_image() {
    apptainer exec --rocm --cleanenv \
        --bind "${BASE}" --bind "${REPO}" \
        --env HF_HOME="${HF_HOME}",OMP_NUM_THREADS=1,HSA_XNACK=1${HF_HUB_OFFLINE:+,HF_HUB_OFFLINE=1}${ROCR_VISIBLE_DEVICES:+,ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES} \
        "${SIF}" bash -c "source /usr/share/lmod/lmod/init/bash; module load ${MAIA_MODULE} >/dev/null 2>&1 || { echo 'module load ${MAIA_MODULE} failed'; exit 1; }; [[ -f ${VENV}/bin/activate ]] && source ${VENV}/bin/activate; $1"
}
