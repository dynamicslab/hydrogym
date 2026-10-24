# m-AIA host file for the ROCm image (rocm/Dockerfile.hydrogym-rocm), selected
# via MAIA_HOST_FILE (see cmake/Configure.cmake) - the AMD counterpart of
# base/DEVCONTAINER.cmake. Compiler: auxiliary/compilers/AMDGPU_AAC6.cmake
# (hipstdpar; AMDGPU.cmake plus the workarounds for ROCm 7's rocPRIM, whose
# device stable_sort rejects m-AIA's comparators), called through an mpicxx wrapper with OMPI_CXX=hipcc
# (build_maia_rocm.sh); libraries built against the image's Open MPI in
# /opt/maia_deps.
set(MAIA_HOST "ROCMCONTAINER")
set(HOST "ROCMCONTAINER")
set(HOST_SUPPORTED_COMPILERS "AMDGPU_AAC6")
set(HOST_DEFAULT_COMPILER "AMDGPU_AAC6")

set(HOST_COMPILER_EXE_AMDGPU_AAC6 "/opt/maia/wrap/mpicxx")

set(INCLUDE_DIRS
    "/opt/maia_deps/parallel-netcdf-1.14.0/include"
    "/opt/maia_deps/fftw-3.3.10/include"
)

set(LIBRARY_DIRS
    "/opt/maia_deps/parallel-netcdf-1.14.0/lib"
    "/opt/maia_deps/fftw-3.3.10/lib"
)

set(LIBRARY_NAMES "m" "pnetcdf" "fftw3_mpi" "fftw3")

if(WITH_HDF5)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} "/opt/maia_deps/hdf5-1.14.5/include")
  set(LIBRARY_DIRS ${LIBRARY_DIRS} "/opt/maia_deps/hdf5-1.14.5/lib")
  set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5_hl" "hdf5" "z" "dl")
endif()

# rocPRIM picks its tuning from this (gfx942 -> 942), see auxiliary/hosts/AAC6.cmake.
set(HOST_CONFIGURE "export LC_ALL=C" "export LANG=C" "export ROCPRIM_TARGET_ARCH=942")
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_PRESERVE_ENV "=")
