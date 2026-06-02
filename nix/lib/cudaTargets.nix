# nix/lib/cudaTargets.nix
#
# GPU architecture target tables consumed by mkBackendShell.
# Each target produces a separate dev-shell output; the closures differ
# in the AOT-compiled CUDA kernels they ship. The split mirrors the
# existing clagemann/hydrogym-nvhpc-26.1_cuda-12.9_{hopper_blackwell,
# turing_ampere}:latest image tags.
{
  hopper-blackwell = {
    name = "hopper-blackwell";
    # nvcc --gpu-architecture flags (compute capability list, comma-separated)
    nvccGencode = "compute_90,compute_100";
    # NVHPC's -gpu= compiler flag
    nvhpcGpuArch = "cc90,cc100";
    # JAX's XLA_FLAGS target list (deviceless AOT compile)
    xlaCudaTargets = "sm_90,sm_100";
  };

  turing-ampere = {
    name = "turing-ampere";
    nvccGencode = "compute_75,compute_80,compute_86";
    nvhpcGpuArch = "cc75,cc80,cc86";
    xlaCudaTargets = "sm_75,sm_80,sm_86";
  };
}
