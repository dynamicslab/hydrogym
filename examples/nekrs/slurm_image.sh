# shellcheck shell=bash disable=SC2034
# Container image of the Slurm scripts (sourced by wing/run_slurm.sbatch, wing/bl_stats_slurm.sbatch and
# channel/train_slurm.sbatch). Sets SIF, BACKEND and GPUFLAG (apptainer --nv | --rocm) and checks that the image
# exists.
#
# BACKEND=CUDA (default): the HydroGym GPU image as an Apptainer SIF, by default in the directory of the job script.
#   x86_64:  apptainer pull docker://clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2
#            (-> hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif, see docs/docs/installation/containers.md)
#   aarch64: hydrogym-gh200.sif, built from .devcontainer/release/apptainer/ (same page, "NVIDIA GH200 nodes")
# BACKEND=HIP: an AMD ROCm image, built with .devcontainer/scripts/build_rocm_image.sh (GPU_ARCH, e.g. gfx942) and
#   converted to a SIF:  podman save -o hydrogym-rocm.tar <TAG>
#                        apptainer build hydrogym-rocm.sif docker-archive://hydrogym-rocm.tar
# SIF=<path> selects another image.
BACKEND="${BACKEND:-CUDA}"
case "$BACKEND" in
    CUDA)
        GPUFLAG=--nv
        if [[ "$(uname -m)" == aarch64 ]]; then
            SIF="${SIF:-$PWD/hydrogym-gh200.sif}"
        else
            SIF="${SIF:-$PWD/hydrogym-nvhpc-26.5_cuda-12.9_1.0.2.sif}"
        fi
        ;;
    HIP)
        GPUFLAG=--rocm
        SIF="${SIF:-$PWD/hydrogym-rocm.sif}"
        ;;
    *)
        echo "BACKEND must be CUDA or HIP, not $BACKEND"
        exit 1
        ;;
esac
if [[ ! -f "$SIF" ]]; then
    echo "image $SIF not found: set SIF=<path to the .sif>, see examples/nekrs/slurm_image.sh"
    exit 1
fi
echo "image $SIF (BACKEND=$BACKEND)"
