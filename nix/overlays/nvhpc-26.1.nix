# nix/overlays/nvhpc-26.1.nix
#
# NVHPC SDK 26.1 (containing CUDA 12.9 and 13.1, NCCL, math libs, nvcc,
# nvfortran, nvc/nvc++) as a fixed-output derivation. This is the largest
# input to any GPU flake output — multi-GB in the store. Build it once per
# host and re-use via the Nix binary cache.

{ stdenv, fetchurl, lib, autoPatchelfHook, zlib, glibc, libGL }:

stdenv.mkDerivation rec {
  pname = "nvhpc-sdk";
  version = "26.1";

  src = fetchurl {
    url = "https://developer.download.nvidia.com/hpc-sdk/${version}/nvhpc_2026_261_Linux_x86_64_cuda_multi.tar.gz";
    hash = "sha256-FkAKKOIUAt3fCxaZqhffU/EBf3ODVdX6DCc6Ndnw3wY=";
  };

  # The tarball's top-level dir wraps install_components/. Strip the wrapper
  # so the install tree lands flat under $out/Linux_x86_64/26.1/...
  sourceRoot = "nvhpc_2026_261_Linux_x86_64_cuda_multi/install_components";

  nativeBuildInputs = [ autoPatchelfHook ];

  # Runtime libs the patched ELFs need to resolve at load time.
  # stdenv.cc.cc.lib supplies libgcc_s.so.1 / libstdc++.so.6, required by
  # the cublas/cusparse/cusolver/cufft/cutensor math libraries.
  buildInputs = [ zlib glibc libGL stdenv.cc.cc.lib ];

  # Nsight Compute (profilers/.../Nsight_Compute) drags in optional Qt UI
  # plugins (libqxcb, libqwayland-*), InfiniBand RDMA libs (libib*), and
  # libnvidia-ml (host driver lib, never in the store). None of these are
  # on the JAX runtime path — let autoPatchelfHook warn instead of error.
  autoPatchelfIgnoreMissingDeps = true;

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -r . "$out/"

    # Symlink the canonical binaries to $out/bin for PATH consumption.
    mkdir -p "$out/bin"
    for tool in nvcc nvc nvc++ nvfortran nvprof nsys ncu; do
      if [ -e "$out/Linux_x86_64/${version}/compilers/bin/$tool" ]; then
        ln -s "../Linux_x86_64/${version}/compilers/bin/$tool" "$out/bin/$tool"
      fi
    done

    runHook postInstall
  '';

  meta = with lib; {
    description = "NVIDIA HPC SDK 26.1 (CUDA 12.9 + 13.1)";
    homepage = "https://developer.nvidia.com/hpc-sdk";
    license = licenses.unfree;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
