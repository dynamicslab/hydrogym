# Shared settings for setup_aac.sh / run_aac.sbatch (sourced, not run).
# Everything (m-AIA hipstdpar build, Python venv with hydrogym[maia], torch) comes from the
# HydroGym ROCm image (.devcontainer/rocm/Dockerfile.hydrogym-rocm); the training script is
# taken from the REPO checkout.
REPO="${REPO:-$HOME/hydrogym}"                  # hydrogym checkout (kl/nekrs or later)
BASE="${BASE:-$HOME/hydrogym-baselines}"        # image, HF cache, runs
SIF="${SIF:-${BASE}/hydrogym-rocm.sif}"
IMAGE="${IMAGE:-docker://docker.io/kl126162/hydrogym-rocm:rocm7.2.4-gfx942}"
export HF_HOME="${BASE}/hf_cache"

# Run "$@" (a bash command string) in the image.
in_image() {
    apptainer exec --rocm --cleanenv \
        --bind "${BASE}" --bind "${REPO}" \
        --env HF_HOME="${HF_HOME}",OMP_NUM_THREADS=1,HSA_XNACK=1${HF_HUB_OFFLINE:+,HF_HUB_OFFLINE=1}${ROCR_VISIBLE_DEVICES:+,ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES} \
        "${SIF}" bash -c "$1"
}
