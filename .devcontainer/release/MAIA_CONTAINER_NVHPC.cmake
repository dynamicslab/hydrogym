# m-AIA host file for the CUDA release image (Dockerfile.hydrogym-cuda),
# selected via MAIA_HOST_FILE (see cmake/Configure.cmake). NVHPC 26.5 nvc++
# with stdpar for CUDA 12.9 (the same CUDA as NekRS), HPC-X Open MPI built for
# CUDA 12.9, libraries from build_maia_deps.sh in /opt/maia_deps. The GPU
# architecture comes from configure.py --enable-pstl (build_maia_container.sh).
set(MAIA_HOST "MAIACONTAINER")
set(HOST "MAIACONTAINER")
set(HOST_SUPPORTED_COMPILERS "NVHPC")
set(HOST_DEFAULT_COMPILER "NVHPC")

# x86_64 images, or aarch64 (GH200: apptainer/hydrogym-cuda-build.def)
if(EXISTS "/opt/nvidia/hpc_sdk/Linux_aarch64/26.5")
  set(_nvhpc "/opt/nvidia/hpc_sdk/Linux_aarch64/26.5")
else()
  set(_nvhpc "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5")
endif()
set(_mpi "${_nvhpc}/comm_libs/12.9/hpcx/hpcx-2.50/ompi")

set(HOST_COMPILER_EXE_NVHPC "${_nvhpc}/compilers/bin/nvc++")

set(INCLUDE_DIRS
    "${_nvhpc}/math_libs/12.9/include"
    "/opt/maia_deps/parallel-netcdf-1.14.0/include"
    "/opt/maia_deps/fftw-3.3.10/include"
    "${_mpi}/include"
)

set(LIBRARY_DIRS
    "${_nvhpc}/math_libs/12.9/lib64"
    "/opt/maia_deps/parallel-netcdf-1.14.0/lib"
    "/opt/maia_deps/fftw-3.3.10/lib"
    "${_mpi}/lib"
)

set(LIBRARY_NAMES "cufftw" "cufft" "m" "pnetcdf" "fftw3_mpi" "fftw3" "mpi")

# No CUDA include directory here: nvc++ -gpu=cuda12.9 adds its own, and an
# explicit one would put CUDA 12.9's CCCL ahead of the newer CCCL that NVHPC's
# stdpar headers are written for.
set(INCLUDE_DIRS_NVHPC
  "${_nvhpc}/compilers/include"
)
set(LIBRARY_DIRS_NVHPC
  "${_nvhpc}/compilers/lib"
  "${_nvhpc}/cuda/12.9/lib64"
)
set(INCLUDE_DIRS_nvhpc ${INCLUDE_DIRS_NVHPC})
set(LIBRARY_DIRS_nvhpc ${LIBRARY_DIRS_NVHPC})

if(WITH_HDF5)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} "/opt/maia_deps/hdf5-1.14.5/include")
  set(LIBRARY_DIRS ${LIBRARY_DIRS} "/opt/maia_deps/hdf5-1.14.5/lib")
  # HDF5's configure finds the SDK image's szip (libaec): sz, aec
  set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5_hl" "hdf5" "sz" "aec" "z" "dl")
endif()

set(CUDA_VERSION "12.9")

set(HOST_CONFIGURE "export LC_ALL=C" "export LANG=C")
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_PRESERVE_ENV "=")
