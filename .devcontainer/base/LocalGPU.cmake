set(MAIA_HOST "LocalGPU")
set(HOST "LocalGPU")
set(HOST_SUPPORTED_COMPILERS "NVHPC")
set(HOST_DEFAULT_COMPILER "NVHPC")

set(HOST_COMPILER_EXE_NVHPC "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/compilers/bin/nvc++")

set(INCLUDE_DIRS
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/include"
    "/opt/maia_deps/parallel-netcdf-1.14.0/include"
    "/opt/maia_deps/fftw-3.3.10/include"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/include"
)

set(LIBRARY_DIRS
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/lib64"
    "/opt/maia_deps/parallel-netcdf-1.14.0/lib"
    "/opt/maia_deps/fftw-3.3.10/lib"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/ompi5/lib"
)

set(LIBRARY_NAMES "cufftw" "cufft" "m" "pnetcdf" "fftw3_mpi" "fftw3" "mpi")

set(INCLUDE_DIRS_NVHPC
  "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/compilers/include"
  "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/include"
)
set(LIBRARY_DIRS_NVHPC
  "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/compilers/lib"
  "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/lib64"
)

set(INCLUDE_DIRS_nvhpc ${INCLUDE_DIRS_NVHPC})
set(LIBRARY_DIRS_nvhpc ${LIBRARY_DIRS_NVHPC})

if(WITH_HDF5)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} "/opt/maia_deps/hdf5-1.14.5/include")
  set(LIBRARY_DIRS ${LIBRARY_DIRS} "/opt/maia_deps/hdf5-1.14.5/lib")
  set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5_hl" "hdf5" "sz" "aec" "z" "dl")
endif()

set(CUDA_VERSION "13.2")
set(PSTL "ada")

set(HOST_CONFIGURE "export LC_ALL=C" "export LANG=C")
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_PRESERVE_ENV "=")
